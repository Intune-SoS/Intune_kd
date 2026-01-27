"""
Standalone Pipeline Trigger Job (Python 3.9 Compatible)
========================================================

This standalone version replaces the Spark streaming job with a simple polling
approach that works with Python 3.9. It reads from Kafka, maintains counts,
and emits triggers when threshold is reached.

This is a temporary compatibility workaround while we resolve Spark/Python
version compatibility. It provides the same functionality as the Spark job
but uses standard Python libraries instead of PySpark.

Architecture:
    Kafka (intune.status.events) → Python Consumer → {
        1. Count records with status_eval_first='done'
        2. Trigger emission to Kafka (pipeline.triggers) when count >= threshold
        3. Count upserts to Supabase (pipeline_status_counts)
    }

Trigger Rule:
    When COUNT(status_eval_first='done') >= 2 (demo threshold):
      → Emit trigger to start finetune workflow
"""

import os
import sys
import json
import logging
import signal
import time
from uuid import uuid4
from datetime import datetime, timezone
from collections import defaultdict
from typing import Dict, Set, Optional
from dotenv import load_dotenv

# Kafka consumer and producer
from confluent_kafka import Consumer, Producer, KafkaError

# Load environment variables
load_dotenv()

# Add project root to path for imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import get_supabase_client, upsert_pipeline_count

# ============================================================================
# CONFIGURATION
# ============================================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_EVENTS = os.getenv("KAFKA_TOPIC_EVENTS", "intune.status.events")
KAFKA_TOPIC_TRIGGERS = os.getenv("KAFKA_TOPIC_TRIGGERS", "pipeline.triggers")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "pipeline-standalone-processor")

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Threshold for trigger emission (demo: 2 rows, production: 5000)
TRIGGER_THRESHOLD = int(os.getenv("TRIGGER_THRESHOLD", "2"))

# Processing interval
PROCESSING_INTERVAL = int(os.getenv("PROCESSING_INTERVAL", "10"))  # seconds

# Trigger logic
TARGET_STATUS = "done"
TRIGGER_STAGE = "finetune_and_evaluate"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global flag for graceful shutdown
running = True

# ============================================================================
# PIPELINE PROCESSOR
# ============================================================================

class StandalonePipelineProcessor:
    """Standalone pipeline processor that replaces Spark streaming."""

    def __init__(self):
        """Initialize Kafka consumer and producer."""
        # Consumer config
        self.consumer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': KAFKA_GROUP_ID,
            'auto.offset.reset': 'latest',
            'enable.auto.commit': False,  # Manual commit for reliability
        }

        # Producer config
        self.producer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'enable.idempotence': True,
            'acks': 'all'
        }

        self.consumer = Consumer(self.consumer_config)
        self.producer = Producer(self.producer_config)

        self.consumer.subscribe([KAFKA_TOPIC_EVENTS])
        logger.info(f"Initialized: consumer group={KAFKA_GROUP_ID}, events topic={KAFKA_TOPIC_EVENTS}")

        # State tracking
        self.status_counts: Dict[str, int] = defaultdict(int)
        self.processed_events: Set[str] = set()  # Deduplication
        self.last_trigger_count = 0  # Track when we last triggered

    def process_event(self, event: Dict) -> bool:
        """
        Process a single event and update counts.

        Args:
            event: Parsed event from Kafka

        Returns:
            bool: True if processed successfully, False otherwise
        """
        try:
            event_id = event.get("event_id")
            status_eval_first = event.get("status_eval_first")

            # Deduplication check
            if event_id in self.processed_events:
                logger.debug(f"Skipping duplicate event: {event_id}")
                return True

            # Track this event
            self.processed_events.add(event_id)

            # Update counts for all status values
            if status_eval_first:
                old_count = self.status_counts[status_eval_first]
                self.status_counts[status_eval_first] += 1
                new_count = self.status_counts[status_eval_first]

                logger.debug(f"Updated count: {status_eval_first} {old_count} -> {new_count}")

            return True

        except Exception as e:
            logger.error(f"Error processing event: {e}")
            return False

    def update_supabase_counts(self):
        """Update pipeline_status_counts table in Supabase."""
        try:
            logger.debug("Updating Supabase counts...")

            for status, count in self.status_counts.items():
                upsert_pipeline_count(checkpoint=-1, status=status, count=count)

            logger.debug(f"Updated {len(self.status_counts)} status counts in Supabase")

        except Exception as e:
            logger.error(f"Failed to update Supabase counts: {e}")

    def check_and_emit_trigger(self):
        """Check if trigger condition is met and emit trigger if needed."""
        try:
            current_done_count = self.status_counts.get(TARGET_STATUS, 0)

            # Only trigger if we've crossed the threshold since last trigger
            if current_done_count >= TRIGGER_THRESHOLD and current_done_count > self.last_trigger_count:
                logger.info(f"Trigger condition met: {TARGET_STATUS}={current_done_count} >= {TRIGGER_THRESHOLD}")

                # Create trigger event
                trigger_event = {
                    "trigger_id": str(uuid4()),
                    "stage": TRIGGER_STAGE,
                    "reason": "threshold_reached",
                    "threshold": TRIGGER_THRESHOLD,
                    "observed_count": current_done_count,
                    "fired_at": datetime.now(timezone.utc).isoformat(),
                    "dedupe_key": f"finetune_{TRIGGER_THRESHOLD}_{current_done_count}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
                    "source_job": "standalone_processor"
                }

                # Publish trigger to Kafka
                key = TRIGGER_STAGE
                value = json.dumps(trigger_event, ensure_ascii=False)

                self.producer.produce(
                    topic=KAFKA_TOPIC_TRIGGERS,
                    key=key.encode('utf-8'),
                    value=value.encode('utf-8')
                )
                self.producer.poll(0)

                # Update trigger tracking
                self.last_trigger_count = current_done_count

                logger.info(f"🚀 Trigger emitted: {trigger_event['dedupe_key']} (count={current_done_count})")

        except Exception as e:
            logger.error(f"Error checking/emitting trigger: {e}")

    def consume_and_process(self):
        """Main consume loop with periodic processing."""
        global running

        logger.info("Starting consume and process loop...")
        last_processing_time = 0

        while running:
            try:
                # Poll for messages (short timeout for responsiveness)
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    # No message - check if it's time for periodic processing
                    current_time = time.time()
                    if current_time - last_processing_time >= PROCESSING_INTERVAL:
                        self.update_supabase_counts()
                        self.check_and_emit_trigger()
                        last_processing_time = current_time

                        # Log current status for monitoring
                        total_events = len(self.processed_events)
                        done_count = self.status_counts.get(TARGET_STATUS, 0)
                        logger.info(f"Status: processed={total_events} events, {TARGET_STATUS}={done_count}, trigger_threshold={TRIGGER_THRESHOLD}")

                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug("Reached end of partition")
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                    continue

                # Parse and process event
                try:
                    value = msg.value().decode('utf-8')
                    event = json.loads(value)

                    success = self.process_event(event)

                    if success:
                        self.consumer.commit(message=msg)
                    else:
                        logger.warning("Event processing failed - offset not committed")

                except Exception as e:
                    logger.error(f"Failed to parse/process message: {e}")

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break

            except Exception as e:
                logger.error(f"Error in consume loop: {e}")
                time.sleep(1)  # Brief pause on errors

        logger.info("Consume loop stopped")

    def close(self):
        """Clean shutdown."""
        logger.info("Shutting down processor...")

        # Final count update
        try:
            self.update_supabase_counts()
        except Exception as e:
            logger.error(f"Error in final count update: {e}")

        # Close Kafka connections
        self.producer.flush(timeout=5)
        self.consumer.close()

        logger.info("Processor shutdown complete")

# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

processor_instance: Optional[StandalonePipelineProcessor] = None

def signal_handler(signum, frame):
    """Handle SIGTERM and SIGINT for graceful shutdown."""
    global running
    logger.info(f"Received signal {signum}, initiating shutdown...")
    running = False

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

# ============================================================================
# MAIN
# ============================================================================

def main():
    """Main entry point."""
    global processor_instance

    logger.info("=" * 60)
    logger.info("Standalone Pipeline Processor (Python 3.9 Compatible)")
    logger.info("=" * 60)
    logger.info(f"Kafka Bootstrap: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Input Topic:     {KAFKA_TOPIC_EVENTS}")
    logger.info(f"Output Topic:    {KAFKA_TOPIC_TRIGGERS}")
    logger.info(f"Threshold:       {TRIGGER_THRESHOLD}")
    logger.info(f"Processing Interval: {PROCESSING_INTERVAL}s")
    logger.info("=" * 60)

    # Validate environment
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

    processor_instance = StandalonePipelineProcessor()

    try:
        processor_instance.consume_and_process()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        if processor_instance:
            processor_instance.close()

if __name__ == "__main__":
    main()