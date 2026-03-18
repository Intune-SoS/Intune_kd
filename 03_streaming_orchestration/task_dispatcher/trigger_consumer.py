"""
Trigger Consumer - Event-Driven Finetune Execution
===================================================

This consumer replaces the polling loop in eval_finetune.py with event-driven execution triggered by Kafka events. It provides:
Instead of checking every 5 minutes "is it time to fine-tune?", it listens for a Kafka message that says "fine-tune now" and then runs the full workflow — prepare data → fine-tune → evaluate.

1. Idempotent trigger execution via deduplication
2. Kafka offset management for exactly-once semantics
3. Manual fallback mode for emergency execution
4. Graceful shutdown with offset commit guarantee

Architecture:
    Kafka (pipeline.triggers) → This Consumer → {
        prepare_training_data() → run_finetune() → evaluate_with_finetuned_model()
    }

Replaces:
    - Polling loop in eval_finetune.py:check_finetune_conditions()
    - Manual threshold checking every 5 minutes
"""

import os
import sys
import signal
import json
import logging
import argparse
from typing import Dict, Callable, Optional
from dotenv import load_dotenv

# Kafka consumer
from confluent_kafka import Consumer, KafkaError, TopicPartition

# Load environment variables
load_dotenv()

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import (
    get_supabase_client,
    insert_trigger_log_if_new,
    mark_event_consumed
)

# ============================================================================
# CONFIGURATION
# ============================================================================

KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")
KAFKA_TOPIC_TRIGGERS = os.getenv("KAFKA_TOPIC_TRIGGERS", "pipeline.triggers")
KAFKA_GROUP_ID = os.getenv("KAFKA_GROUP_ID", "pipeline-trigger-consumer")

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
# STAGE HANDLERS
# ============================================================================

def run_finetune_and_evaluate() -> None:
    """
    Execute complete finetune workflow in ONE CALL (no polling):
      1. Prepare training data from intune_db (status_eval_first='done')
      2. Run finetuning process
      3. Evaluate ALL pending records with finetuned model

    This replaces the polling loops in eval_finetune.py.
    """
    logger.info("Executing stage=finetune_and_evaluate")

    # Import eval_finetune functions
    import sys
    sys.path.insert(0, os.path.join(project_root, 'app'))

    try:
        from eval_finetune import (
            prepare_training_data,
            run_finetune,
            evaluate_with_finetuned_model
        )

        logger.info("Step 1/3: Preparing training data...")
        if not prepare_training_data():
            raise Exception("Failed to prepare training data")

        logger.info("Step 2/3: Running finetune...")
        if not run_finetune():
            raise Exception("Finetuning failed")

        logger.info("Step 3/3: Evaluating with finetuned model...")
        if not evaluate_with_finetuned_model():
            raise Exception("Post-finetune evaluation failed")

        logger.info("✅ Complete finetune workflow finished successfully")

    except Exception as e:
        logger.error(f"❌ Finetune workflow failed: {e}", exc_info=True)
        raise  # Re-raise to prevent offset commit


# ============================================================================
# STAGE DISPATCH MAPPING
# ============================================================================

STAGE_HANDLERS: Dict[str, Callable[[], None]] = {
    "finetune_and_evaluate": run_finetune_and_evaluate,
}


# ============================================================================
# TRIGGER PROCESSOR
# ============================================================================

class TriggerConsumer:
    """Kafka consumer for pipeline triggers with idempotent execution."""

    def __init__(self):
        """Initialize Kafka consumer."""
        self.config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': KAFKA_GROUP_ID,
            'auto.offset.reset': 'earliest',
            'enable.auto.commit': False,  # Manual commit for exactly-once
        }
        self.consumer = Consumer(self.config)
        self.consumer.subscribe([KAFKA_TOPIC_TRIGGERS])
        logger.info(f"Consumer initialized: group={KAFKA_GROUP_ID} topic={KAFKA_TOPIC_TRIGGERS}")

    def process_trigger(self, trigger_event: Dict) -> bool:
        """
        Process a single trigger event with deduplication.

        Args:
            trigger_event: Parsed trigger event dictionary

        Returns:
            bool: True if processed successfully, False otherwise
        """
        try:
            # Extract fields
            trigger_id = trigger_event.get("trigger_id")
            stage = trigger_event.get("stage")
            dedupe_key = trigger_event.get("dedupe_key")

            if not all([trigger_id, stage, dedupe_key]):
                logger.warning(f"Incomplete trigger event: {trigger_event}")
                return False

            logger.info(f"Processing trigger: {dedupe_key}")
            import time
            print(f"\n[METRIC 2] TRIGGER RECEIVED at: {time.time()}") # ADD THIS

            # Custom deduplication using the trigger's dedupe_key
            try:
                client = get_supabase_client()

                # Check if this dedupe_key was already processed
                existing = client.table("pipeline_trigger_log")\
                    .select("trigger_id")\
                    .eq("dedupe_key", dedupe_key)\
                    .execute()

                if existing.data:
                    logger.info(f"Skipping duplicate trigger: {dedupe_key}")
                    return True  # Already processed

                # Insert new trigger log entry with proper UUID
                log_data = {
                    "trigger_id": trigger_id,  # Use UUID from trigger
                    "checkpoint": -1,
                    "stage": stage,
                    "dedupe_key": dedupe_key,  # Use custom dedupe_key from trigger
                    "source_job": trigger_event.get("source_job", "trigger_consumer")
                }

                client.table("pipeline_trigger_log").insert(log_data).execute()
                logger.info(f"Logged new trigger: {dedupe_key}")

            except Exception as e:
                logger.error(f"Error in deduplication logic: {e}")
                return False

            # Dispatch to stage handler
            handler = STAGE_HANDLERS.get(stage)

            if handler is None:
                logger.warning(f"Unknown stage: {stage} - skipping")
                return True  # Skip unknown stages

           # Execute stage (no arguments needed)
            logger.info(f"🚀 Dispatching to handler: {stage}")
            handler()

            # Mark event as consumed
            mark_event_consumed(trigger_id, "trigger_consumer")

            logger.info(f"✅ Trigger processed successfully: {dedupe_key}")
            return True

        except Exception as e:
            logger.error(f"Error processing trigger: {e}", exc_info=True)
            return False

    def consume_loop(self):
        """Main consume loop."""
        global running

        logger.info("Starting consume loop...")

        while running:
            try:
                # Poll for messages (1 second timeout)
                msg = self.consumer.poll(timeout=1.0)

                if msg is None:
                    continue

                if msg.error():
                    if msg.error().code() == KafkaError._PARTITION_EOF:
                        logger.debug("Reached end of partition")
                    else:
                        logger.error(f"Kafka error: {msg.error()}")
                    continue

                # Parse message
                try:
                    value = msg.value().decode('utf-8')
                    trigger_event = json.loads(value)
                except Exception as e:
                    logger.error(f"Failed to parse message: {e}")
                    continue

                # Process trigger
                success = self.process_trigger(trigger_event)

                if success:
                    # Commit offset only after successful processing
                    self.consumer.commit(message=msg)
                    logger.debug(f"Offset committed: partition={msg.partition()} offset={msg.offset()}")
                else:
                    # Do NOT commit offset - allow retry on next poll
                    logger.warning("Trigger processing failed - offset NOT committed (will retry)")

            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                break
            except Exception as e:
                logger.error(f"Error in consume loop: {e}", exc_info=True)

        logger.info("Consume loop stopped")

    def close(self):
        """Close consumer cleanly."""
        logger.info("Closing Kafka consumer...")
        self.consumer.close()
        logger.info("Kafka consumer closed")


# ============================================================================
# MANUAL FALLBACK MODE
# ============================================================================

def manual_execution(stage: str):
    """
    Manual fallback mode: execute stage directly without Kafka.

    Args:
        stage: Stage name
    """
    logger.info("=" * 60)
    logger.info("MANUAL FALLBACK MODE")
    logger.info("=" * 60)
    logger.info(f"Stage: {stage}")
    logger.info("=" * 60)

    handler = STAGE_HANDLERS.get(stage)

    if handler is None:
        logger.error(f"Unknown stage: {stage}")
        sys.exit(1)

    try:
        handler()
        logger.info(f"✅ Manual execution completed: {stage}")
    except Exception as e:
        logger.error(f"❌ Manual execution failed: {e}", exc_info=True)
        sys.exit(1)


# ============================================================================
# SIGNAL HANDLERS
# ============================================================================

consumer_instance: Optional[TriggerConsumer] = None


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
    global consumer_instance

    parser = argparse.ArgumentParser(description='Pipeline Trigger Consumer')
    parser.add_argument('--manual', action='store_true',
                        help='Manual fallback mode (bypass Kafka)')
    parser.add_argument('--stage', type=str,
                        choices=list(STAGE_HANDLERS.keys()),
                        help='Stage to execute (manual mode only)')
    args = parser.parse_args()

    # Manual fallback mode
    if args.manual:
        if not args.stage:
            parser.error("--manual requires --stage")

        manual_execution(args.stage)
        return

    # Event-driven mode (normal operation)
    logger.info("=" * 60)
    logger.info("Pipeline Trigger Consumer (Event-Driven Mode)")
    logger.info("=" * 60)
    logger.info(f"Kafka:  {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Topic:  {KAFKA_TOPIC_TRIGGERS}")
    logger.info(f"Group:  {KAFKA_GROUP_ID}")
    logger.info("=" * 60)

    consumer_instance = TriggerConsumer()

    try:
        consumer_instance.consume_loop()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Trigger consumer shutting down.")
        consumer_instance.close()
        logger.info("Shutdown complete")


if __name__ == "__main__":
    main()
