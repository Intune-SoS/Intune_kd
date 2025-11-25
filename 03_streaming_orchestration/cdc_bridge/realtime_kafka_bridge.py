"""
Supabase Realtime to Kafka Bridge Service
==========================================

This service subscribes to Supabase Realtime and publishes normalized events
to Kafka. It bridges the database change data capture (CDC) system with the
event-driven pipeline architecture.

Purpose:
    - Listen to intune_db table changes (INSERT, UPDATE on status columns)
    - Track both status_eval_first and status_eval_final columns
    - Normalize events into structured JSON format
    - Publish to Kafka topic "intune.status.events"
    - Handle delivery failures with dead-letter queue

Architecture:
    Supabase Realtime (CDC) → This Bridge → Kafka → Spark Consumer

Event Flow:
    1. Record status changes in intune_db table (status_eval_first or status_eval_final)
    2. Supabase Realtime emits change notification
    3. Bridge normalizes event and publishes to Kafka
    4. Downstream consumers process events for trigger logic
"""

import os
import sys
import signal
import json
import logging
from uuid import uuid4
from datetime import datetime, timezone
from typing import Dict, Any, Optional
import asyncio
from dotenv import load_dotenv

# Kafka producer
from confluent_kafka import Producer

# Supabase Realtime client (async)
from supabase import acreate_client, AsyncClient

# Load environment variables
load_dotenv()

# ============================================================================
# CONFIGURATION
# ============================================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_EVENTS = os.getenv("KAFKA_TOPIC_EVENTS", "intune.status.events")
KAFKA_TOPIC_DLQ = os.getenv("KAFKA_TOPIC_DLQ", "pipeline.dlq")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Realtime subscription configuration
REALTIME_TABLE = os.getenv("REALTIME_TABLE", "intune_db")
REALTIME_SCHEMA = os.getenv("REALTIME_SCHEMA", "public")

# Logging configuration
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRY_ATTEMPTS = 3
RETRY_BACKOFF_BASE = 2  # seconds
RETRY_BACKOFF_MAX = 60  # seconds

# ============================================================================
# KAFKA PRODUCER
# ============================================================================

class KafkaEventProducer:
    """Idempotent Kafka producer with dead-letter queue support."""

    def __init__(self):
        """Initialize Kafka producer with idempotent settings."""
        self.config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'enable.idempotence': True,  # Exactly-once semantics
            'acks': 'all',
            'retries': 3,
            'max.in.flight.requests.per.connection': 5,
            'compression.type': 'snappy',
        }
        self.producer = Producer(self.config)
        logger.info(f"Kafka producer initialized: {KAFKA_BOOTSTRAP_SERVERS}")

    def delivery_callback(self, err, msg):
        """Callback for delivery reports."""
        if err:
            logger.error(f"Message delivery failed: {err}")
        else:
            logger.debug(f"Message delivered: topic={msg.topic()} partition={msg.partition()} offset={msg.offset()}")

    def publish_event(self, event: Dict[str, Any], topic: str = KAFKA_TOPIC_EVENTS) -> bool:
        """
        Publish event to Kafka topic.

        Args:
            event: Event dictionary with all required fields
            topic: Target Kafka topic

        Returns:
            bool: True if published successfully, False otherwise
        """
        try:
            key = str(event.get("record_id", ""))
            value = json.dumps(event, ensure_ascii=False)

            self.producer.produce(
                topic=topic,
                key=key.encode('utf-8'),
                value=value.encode('utf-8'),
                callback=self.delivery_callback
            )
            self.producer.poll(0)  # Trigger delivery reports
            return True

        except Exception as e:
            logger.error(f"Failed to publish event: {e}")
            return False

    def publish_to_dlq(self, event: Dict[str, Any], error_reason: str):
        """
        Publish failed event to dead-letter queue.

        Args:
            event: Original event that failed
            error_reason: Description of the failure
        """
        try:
            dlq_event = {
                **event,
                "dlq_reason": error_reason,
                "dlq_timestamp": datetime.now(timezone.utc).isoformat()
            }
            self.publish_event(dlq_event, topic=KAFKA_TOPIC_DLQ)
            logger.warning(f"Event sent to DLQ: {event.get('event_id')} - {error_reason}")

        except Exception as e:
            logger.error(f"Failed to publish to DLQ: {e}")

    def flush(self):
        """Flush pending messages (for graceful shutdown)."""
        logger.info("Flushing Kafka producer...")
        self.producer.flush(timeout=10)
        logger.info("Kafka producer flushed")


# ============================================================================
# REALTIME EVENT HANDLER
# ============================================================================

class RealtimeKafkaBridge:
    """Bridges Supabase Realtime to Kafka."""

    def __init__(self):
        """Initialize bridge with Supabase and Kafka connections."""
        if not SUPABASE_URL or not SUPABASE_KEY:
            raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set in environment")

        # Note: Supabase async client will be initialized in async context
        self.supabase: AsyncClient = None
        self.producer = KafkaEventProducer()
        self.running = True

        logger.info(f"Bridge initialized: table={REALTIME_TABLE} topic={KAFKA_TOPIC_EVENTS}")

    def normalize_event(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Normalize Supabase Realtime payload into structured event format.

        Args:
            payload: Supabase Realtime change payload

        Returns:
            dict: Normalized event with standard fields
        """
        # Extract data from Supabase Realtime payload structure
        data = payload.get("data", {})
        event_type = data.get("type", "UPDATE")  # INSERT or UPDATE
        record = data.get("record", {})
        old_record = data.get("old_record", {})

        # Extract key fields from record
        record_id = record.get("id")

        # Debug logging to verify data extraction
        logger.debug(f"Extracted: record_id={record_id}, event_type={event_type}")
        logger.debug(f"Record keys: {list(record.keys())}")

        # Track both status columns for intune_db workflow
        status_eval_first = record.get("status_eval_first")
        old_status_eval_first = old_record.get("status_eval_first") if old_record else None

        status_eval_final = record.get("status_eval_final")
        old_status_eval_final = old_record.get("status_eval_final") if old_record else None

        # Build normalized event
        event = {
            "event_id": str(uuid4()),
            "source": REALTIME_TABLE,
            "op": event_type,
            "record_id": record_id,
            "status_eval_first": status_eval_first,
            "old_status_eval_first": old_status_eval_first,
            "status_eval_final": status_eval_final,
            "old_status_eval_final": old_status_eval_final,
            "event_ts": datetime.now(timezone.utc).isoformat(),
            "trace_id": None,  # Reserved for future distributed tracing
            "version": "1.0"
        }

        return event

    def handle_change(self, payload: Dict[str, Any]):
        """
        Handle Supabase Realtime change event.

        Args:
            payload: Realtime change payload
        """
        try:
            logger.debug(f"Received change: {payload.get('eventType')}")

            # Normalize event
            event = self.normalize_event(payload)

            # Log the full normalized event for debugging
            logger.info(f"Normalized event: record_id={event.get('record_id')}, status_first={event.get('status_eval_first')}, status_final={event.get('status_eval_final')}")

            # Validate required fields - need at least a record_id
            if not event.get("record_id"):
                logger.warning(f"Skipping event with missing record_id: {event}")
                return

            # Skip events where no status fields are present (but less strict validation)
            if not event.get("status_eval_first") and not event.get("status_eval_final") and not event.get("old_status_eval_first") and not event.get("old_status_eval_final"):
                logger.debug(f"Skipping event with no status changes: record_id={event.get('record_id')}")
                return

            # Publish to Kafka with retry logic
            success = False
            for attempt in range(MAX_RETRY_ATTEMPTS):
                if self.producer.publish_event(event):
                    success = True
                    import time
                    print(f"\n[METRIC 1] DB WRITE DETECTED at: {time.time()}") # ADD THIS
                    status_msg = f"status_eval_first={event['status_eval_first']}, status_eval_final={event['status_eval_final']}"
                    logger.info(f"Published event: record_id={event['record_id']} {status_msg}")
                    break

                # Simple sleep for retry (not async)
                wait_time = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
                logger.warning(f"Retry {attempt + 1}/{MAX_RETRY_ATTEMPTS} after {wait_time}s...")
                import time
                time.sleep(wait_time)

            # Send to DLQ if all retries failed
            if not success:
                self.producer.publish_to_dlq(event, f"Failed after {MAX_RETRY_ATTEMPTS} attempts")

        except Exception as e:
            logger.error(f"Error handling change: {e}", exc_info=True)

    async def subscribe_realtime(self):
        """
        Subscribe to Supabase Realtime with exponential backoff retry.
        """
        attempt = 0
        while self.running:
            try:
                logger.info(f"Subscribing to Realtime: {REALTIME_SCHEMA}.{REALTIME_TABLE}")

                # Initialize async Supabase client
                if self.supabase is None:
                    self.supabase = await acreate_client(SUPABASE_URL, SUPABASE_KEY)
                    logger.info("Async Supabase client initialized")

                # Subscribe to table changes
                channel = self.supabase.channel('intune-db-changes')
                await channel.on_postgres_changes(
                    event='*',  # Listen to INSERT and UPDATE
                    schema=REALTIME_SCHEMA,
                    table=REALTIME_TABLE,
                    callback=self.handle_change
                ).subscribe()

                logger.info("Bridge started. Listening on Supabase Realtime.")
                attempt = 0  # Reset retry counter on success

                # Keep connection alive
                while self.running:
                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"Realtime connection error: {e}", exc_info=True)

                # Exponential backoff
                wait_time = min(RETRY_BACKOFF_BASE ** attempt, RETRY_BACKOFF_MAX)
                logger.warning(f"Reconnecting in {wait_time}s... (attempt {attempt + 1})")
                await asyncio.sleep(wait_time)
                attempt += 1

    async def shutdown(self):
        """Graceful shutdown: flush Kafka producer and stop listening."""
        logger.info("Shutting down bridge...")
        self.running = False
        self.producer.flush()
        if self.supabase:
            await self.supabase.aclose()
        logger.info("Bridge shutdown complete")


# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

bridge_instance: Optional[RealtimeKafkaBridge] = None


def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT for graceful shutdown."""
    logger.info(f"Received signal {signum}, initiating shutdown...")
    if bridge_instance:
        bridge_instance.running = False
    # Let the main loop handle async shutdown


signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)


# ============================================================================
# MAIN
# ============================================================================

async def main():
    """Main entry point for the bridge service."""
    global bridge_instance

    logger.info("=" * 60)
    logger.info("Supabase Realtime → Kafka Bridge Service")
    logger.info("=" * 60)
    logger.info(f"Table: {REALTIME_SCHEMA}.{REALTIME_TABLE}")
    logger.info(f"Kafka: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Topic: {KAFKA_TOPIC_EVENTS}")
    logger.info(f"DLQ:   {KAFKA_TOPIC_DLQ}")
    logger.info("=" * 60)

    bridge_instance = RealtimeKafkaBridge()

    try:
        await bridge_instance.subscribe_realtime()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        await bridge_instance.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
