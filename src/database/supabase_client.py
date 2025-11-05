"""
Supabase database client and utility functions.
"""

import os
from dotenv import load_dotenv
from supabase import create_client, Client

# Load environment variables
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


def get_supabase_client() -> Client:
    """
    Create and return Supabase client.
    
    Returns:
        Client: Initialized Supabase client
        
    Raises:
        ValueError: If SUPABASE_URL or SUPABASE_KEY are not set
    """
    # Prefer server-side service role key if available (allows bypassing RLS for trusted backend)
    key_to_use = SUPABASE_SERVICE_ROLE_KEY or SUPABASE_KEY

    if not SUPABASE_URL or not key_to_use:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY (or SUPABASE_SERVICE_ROLE_KEY) must be set in .env file")

    # If using the anon/public key and your table has Row-Level Security (RLS) enabled,
    # attempts to insert/update rows may fail with a RLS policy error. For server-side
    # inserts from a trusted backend, provide the service role key in SUPABASE_SERVICE_ROLE_KEY.
    return create_client(SUPABASE_URL, key_to_use)


def int8_to_decimal(value):
    """
    Convert INT8 metric value to decimal.
    
    Supabase stores metrics as INT8 (scaled by 10000) to avoid precision issues.
    This function converts them back to float.
    
    Args:
        value: INT8 value or None
        
    Returns:
        float: Decimal value between 0 and 1, or 0.0 if value is None
    """
    if value is None:
        return 0.0
    return round(value / 10000, 4)


def decimal_to_int8(value: float) -> int:
    """
    Convert decimal metric value to INT8 for storage.

    Args:
        value: Float value between 0 and 1

    Returns:
        int: INT8 value scaled by 10000
    """
    if value is None:
        return 0
    return int(round(value * 10000))


# ============================================================================
# EVENT-DRIVEN ARCHITECTURE HELPER METHODS
# ============================================================================
# These methods support idempotent database writes for the Kafka + Spark
# migration, ensuring exactly-once semantics and replay safety.
# ============================================================================

import logging

logger = logging.getLogger(__name__)


def upsert_pipeline_count(checkpoint: int, status: str, count: int) -> None:
    """
    Upsert a rolling count into pipeline_status_counts table.

    This method is called by the Spark consumer to maintain live counts
    of records per (checkpoint, status) combination. On conflict, it updates
    both the count and updated_at timestamp.

    Args:
        checkpoint: Checkpoint number (1-10)
        status: Status value (score, finetune, output_tuned, score_tuned, completed)
        count: Live count of records with this checkpoint and status

    Raises:
        Exception: If Supabase operation fails

    Example:
        >>> upsert_pipeline_count(1, "finetune", 5000)
        # Upserts count: checkpoint=1 status=finetune count=5000
    """
    try:
        client = get_supabase_client()

        data = {
            "checkpoint": checkpoint,
            "status": status,
            "count": count,
            "updated_at": "NOW()"  # PostgreSQL function
        }

        # Upsert: insert or update on conflict (checkpoint, status)
        response = client.table("pipeline_status_counts")\
            .upsert(data, on_conflict="checkpoint,status")\
            .execute()

        logger.debug(f"Upserted count: checkpoint={checkpoint} status={status} count={count}")

    except Exception as e:
        error_msg = f"Failed to upsert pipeline count for checkpoint={checkpoint} status={status}: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def insert_trigger_log_if_new(trigger_id: str, checkpoint: int, stage: str, source_job: str = None) -> bool:
    """
    Insert a trigger log entry if dedupe_key does not already exist.

    This method implements exactly-once trigger execution by checking for duplicate
    dedupe_key values. If the key already exists, the function returns False to skip
    re-execution. This prevents duplicate stage runs caused by Kafka redelivery or replay.

    Args:
        trigger_id: UUID string for this trigger event
        checkpoint: Checkpoint number (1-10)
        stage: Stage name (score, finetune, output_tuned, score_tuned, completed)
        source_job: Optional identifier for the job that emitted this trigger

    Returns:
        bool: True if trigger was newly logged (execute stage), False if already exists (skip)

    Example:
        >>> if insert_trigger_log_if_new("uuid-123", 1, "finetune"):
        ...     run_finetune_stage(1)
        ... else:
        ...     print("Trigger already executed, skipping")
    """
    try:
        client = get_supabase_client()

        # Construct dedupe_key as specified: "{checkpoint}_{stage}_5000"
        dedupe_key = f"{checkpoint}_{stage}_5000"

        data = {
            "trigger_id": trigger_id,
            "checkpoint": checkpoint,
            "stage": stage,
            "dedupe_key": dedupe_key,
            "source_job": source_job
        }

        # Attempt insert
        response = client.table("pipeline_trigger_log").insert(data).execute()

        logger.info(f"Trigger logged: {dedupe_key}")
        return True

    except Exception as e:
        error_str = str(e).lower()

        # Check if error is due to unique constraint violation on dedupe_key
        if "unique" in error_str or "duplicate" in error_str or "constraint" in error_str:
            logger.info(f"Trigger already exists, skipping: {dedupe_key}")
            return False

        # Other errors should be raised
        error_msg = f"Failed to insert trigger log for {dedupe_key}: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e


def mark_event_consumed(event_id: str, consumer_name: str) -> None:
    """
    Mark a Kafka event as consumed for replay safety.

    This method tracks which Kafka events have been fully processed. If the system
    crashes and Kafka rewinds to an earlier offset, already-processed events can be
    skipped by checking this table. Idempotent: silently ignores if event_id already exists.

    Args:
        event_id: UUID string from Kafka event payload
        consumer_name: Identifier for the consumer service (e.g., "trigger_consumer")

    Raises:
        Exception: If Supabase operation fails (excluding primary key conflicts)

    Example:
        >>> mark_event_consumed("event-uuid-456", "trigger_consumer")
        # Event marked consumed: event_id=event-uuid-456 consumer=trigger_consumer
    """
    try:
        client = get_supabase_client()

        data = {
            "event_id": event_id,
            "consumer_name": consumer_name
        }

        # Insert
        response = client.table("pipeline_consumed_events").insert(data).execute()

        logger.info(f"Event marked consumed: event_id={event_id} consumer={consumer_name}")

    except Exception as e:
        error_str = str(e).lower()

        # Silently ignore primary key conflicts (idempotent behavior)
        if "unique" in error_str or "duplicate" in error_str or "primary key" in error_str:
            logger.debug(f"Event already marked consumed (idempotent): event_id={event_id}")
            return

        # Other errors should be raised
        error_msg = f"Failed to mark event consumed for event_id={event_id}: {e}"
        logger.error(error_msg)
        raise Exception(error_msg) from e
