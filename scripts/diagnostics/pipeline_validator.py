"""
Pipeline Validator - Shadow Mode Validation
============================================

This utility compares stream-based counts from pipeline_status_counts (written
by Spark) against direct Supabase count queries to validate the new event-driven
architecture before cutting over from the old polling workers.

Purpose: Shadow Mode Validation for intune_db workflow
    - Old polling worker (eval_finetune.py) continues running
    - New Kafka + Spark system runs alongside
    - This validator runs on schedule or manually to detect discrepancies

Exit Codes:
    0 = All counts match (success)
    1 = Mismatches or missing entries detected (alert required)

Usage:
    # Validate all statuses for intune_db
    python event_driven_pipeline/pipeline_validator.py

    # Validate specific status
    python event_driven_pipeline/pipeline_validator.py --status done

Scheduling (Recommended):
    Run via cron every 5 minutes during shadow mode:

    */5 * * * * cd /path/to/project && \
                source venv/bin/activate && \
                python event_driven_pipeline/pipeline_validator.py || \
                curl -X POST https://hooks.slack.com/... \
                     -d '{"text":"intune_db count mismatch detected!"}'

    This sends a Slack alert on exit code 1 (mismatch detected).
"""

import os
import sys
import logging
import argparse
from typing import Dict, Tuple, Optional
from collections import defaultdict
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import get_supabase_client

# ============================================================================
# CONFIGURATION
# ============================================================================

# Source table for direct count queries
SOURCE_TABLE = os.getenv("SOURCE_TABLE", "intune_db")
STATUS_COLUMN = os.getenv("STATUS_COLUMN", "status_eval_first")

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# COUNT FETCHERS
# ============================================================================


def fetch_stream_counts(
    status_filter: Optional[str] = None
) -> Dict[str, int]:
    """
    Fetch counts from pipeline_status_counts table (written by Spark).

    Note: intune_db workflow uses checkpoint=-1 as sentinel (no checkpoints).

    Args:
        status_filter: Optional status value to filter by

    Returns:
        dict: {status: count}
    """
    try:
        logger.info("Fetching stream counts from pipeline_status_counts...")

        client = get_supabase_client()
        query = client.table("pipeline_status_counts")\
            .select("checkpoint, status, count")\
            .eq("checkpoint", -1)  # Filter for intune_db entries

        # Apply status filter
        if status_filter is not None:
            query = query.eq("status", status_filter)

        response = query.execute()

        counts = {}
        for row in response.data:
            status = row["status"]
            counts[status] = row["count"]

        logger.info(f"Fetched {len(counts)} stream count entries")
        return counts

    except Exception as e:
        logger.error(f"Failed to fetch stream counts: {e}", exc_info=True)
        raise


def fetch_direct_counts(
    status_filter: Optional[str] = None
) -> Dict[str, int]:
    """
    Fetch counts directly from source table (intune_db).

    Args:
        status_filter: Optional status value to filter by

    Returns:
        dict: {status: count}
    """
    try:
        logger.info(f"Fetching direct counts from {SOURCE_TABLE}...")

        client = get_supabase_client()

        # Fetch all records and aggregate in Python
        query = client.table(SOURCE_TABLE).select(f"{STATUS_COLUMN}")

        # Apply status filter
        if status_filter is not None:
            query = query.eq(STATUS_COLUMN, status_filter)

        response = query.execute()

        # Aggregate counts in Python
        counts = defaultdict(int)
        for row in response.data:
            status = row.get(STATUS_COLUMN)
            if status is not None:
                counts[status] += 1

        logger.info(f"Fetched and aggregated {len(counts)} direct count entries")
        return dict(counts)

    except Exception as e:
        logger.error(f"Failed to fetch direct counts: {e}", exc_info=True)
        raise


# ============================================================================
# COMPARISON LOGIC
# ============================================================================


def compare_counts(
    stream_counts: Dict[Tuple[int, str], int],
    direct_counts: Dict[Tuple[int, str], int]
) -> Tuple[int, int, int]:
    """
    Compare stream and direct counts, logging discrepancies.

    Args:
        stream_counts: Counts from pipeline_status_counts
        direct_counts: Counts from direct table query

    Returns:
        tuple: (ok_count, mismatch_count, missing_count)
    """
    ok_count = 0
    mismatch_count = 0
    missing_count = 0

    # Get all unique keys
    all_keys = set(stream_counts.keys()) | set(direct_counts.keys())

    logger.info("")
    logger.info("=" * 70)
    logger.info("COMPARISON RESULTS")
    logger.info("=" * 70)

    for key in sorted(all_keys):
        checkpoint, status = key
        stream_count = stream_counts.get(key, 0)
        direct_count = direct_counts.get(key, 0)

        if stream_count == direct_count:
            logger.info(f"✅ OK: checkpoint={checkpoint} status={status:15} count={direct_count}")
            ok_count += 1
        elif key not in stream_counts:
            logger.warning(f"⚠️  MISSING FROM STREAM: checkpoint={checkpoint} status={status:15} direct={direct_count}")
            missing_count += 1
        else:
            delta = stream_count - direct_count
            logger.warning(
                f"❌ MISMATCH: checkpoint={checkpoint} status={status:15} "
                f"stream={stream_count} direct={direct_count} delta={delta:+d}"
            )
            mismatch_count += 1

    # Summary
    total = len(all_keys)
    logger.info("=" * 70)
    logger.info("SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total checks:    {total}")
    logger.info(f"OK:              {ok_count}")
    logger.info(f"Mismatches:      {mismatch_count}")
    logger.info(f"Missing:         {missing_count}")
    logger.info("=" * 70)

    return ok_count, mismatch_count, missing_count


# ============================================================================
# MAIN
# ============================================================================


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Pipeline Count Validator (Shadow Mode)')
    parser.add_argument('--checkpoint', type=int, help='Filter by checkpoint number (1-10)')
    parser.add_argument('--status', type=str, help='Filter by status value')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Pipeline Validator - Shadow Mode")
    logger.info("=" * 70)
    logger.info(f"Source Table:  {SOURCE_TABLE}")
    logger.info(f"Status Column: {STATUS_COLUMN}")

    if args.checkpoint:
        logger.info(f"Filter:        checkpoint={args.checkpoint}")
    if args.status:
        logger.info(f"Filter:        status={args.status}")

    logger.info("=" * 70)

    try:
        # Fetch counts from both sources
        stream_counts = fetch_stream_counts(args.checkpoint, args.status)
        direct_counts = fetch_direct_counts(args.checkpoint, args.status)

        # Compare
        ok_count, mismatch_count, missing_count = compare_counts(stream_counts, direct_counts)

        # Exit code
        if mismatch_count == 0 and missing_count == 0:
            logger.info("")
            logger.info("✅ VALIDATION PASSED: All counts match!")
            sys.exit(0)
        else:
            logger.error("")
            logger.error("❌ VALIDATION FAILED: Discrepancies detected!")
            sys.exit(1)

    except Exception as e:
        logger.error(f"Validation error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
