"""
Spark Structured Streaming Pipeline Trigger Job
===============================================

This Spark job reads status events from Kafka, maintains running counts of
records with status_eval_first='done', and emits a trigger when threshold is reached.

Architecture:
    Kafka (intune.status.events) → Spark Streaming → {
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
from uuid import uuid4
from datetime import datetime, timezone
from typing import Iterator, Tuple
from dataclasses import dataclass
from dotenv import load_dotenv

# PySpark imports
from pyspark.sql import SparkSession
from pyspark.sql.functions import (
    from_json, col, struct, to_json, window, current_timestamp
)
from pyspark.sql.types import (
    StructType, StructField, StringType, IntegerType, TimestampType
)
from pyspark.sql.streaming.state import GroupState, GroupStateTimeout

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

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

# Spark checkpoint directory (local or cloud path)
SPARK_CHECKPOINT_DIR = os.getenv("SPARK_CHECKPOINT_DIR", "/tmp/spark-checkpoints/pipeline")

# Threshold for trigger emission (demo: 2 rows, production: 5000)
TRIGGER_THRESHOLD = int(os.getenv("TRIGGER_THRESHOLD", "2"))

# Stall detection timeout (minutes)
STALL_TIMEOUT_MINUTES = int(os.getenv("STALL_TIMEOUT_MINUTES", "30"))

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ============================================================================
# SCHEMA DEFINITIONS
# ============================================================================

# Kafka event schema (from realtime_kafka_bridge.py)
event_schema = StructType([
    StructField("event_id", StringType(), False),
    StructField("source", StringType(), False),
    StructField("op", StringType(), False),
    StructField("record_id", IntegerType(), True),
    StructField("status_eval_first", StringType(), True),
    StructField("old_status_eval_first", StringType(), True),
    StructField("status_eval_final", StringType(), True),
    StructField("old_status_eval_final", StringType(), True),
    StructField("event_ts", StringType(), False),
    StructField("trace_id", StringType(), True),
    StructField("version", StringType(), False)
])

# ============================================================================
# TRIGGER LOGIC
# ============================================================================

# Trigger when status_eval_first changes to 'done'
TARGET_STATUS = "score"
TRIGGER_STAGE = "finetune"

# ============================================================================
# SUPABASE COUNT WRITE
# ============================================================================

def write_counts_to_supabase(batch_df, batch_id):
    """
    Write aggregated counts to Supabase pipeline_status_counts table.

    Args:
        batch_df: DataFrame with (status_eval_first, count) columns
        batch_id: Spark micro-batch ID
    """
    try:
        logger.info(f"Writing counts to Supabase (batch {batch_id})...")

        # Collect rows from DataFrame
        rows = batch_df.collect()

        for row in rows:
            status = row["status_eval_first"]
            count = row["count"]

            try:
                # Use checkpoint=-1 to indicate "global count" (no checkpoints in intune_db)
                upsert_pipeline_count(checkpoint=-1, status=status, count=count)
            except Exception as e:
                logger.error(f"Failed to upsert count for status={status}: {e}")

        logger.info(f"Batch {batch_id} written to Supabase ({len(rows)} rows)")

    except Exception as e:
        logger.error(f"Error in foreachBatch (batch {batch_id}): {e}", exc_info=True)


# ============================================================================
# SPARK JOB
# ============================================================================

def main():
    """Main Spark Structured Streaming job."""

    logger.info("=" * 60)
    logger.info("INTUNE Pipeline Trigger Job (Spark Structured Streaming)")
    logger.info("=" * 60)
    logger.info(f"Kafka Bootstrap: {KAFKA_BOOTSTRAP_SERVERS}")
    logger.info(f"Input Topic:     {KAFKA_TOPIC_EVENTS}")
    logger.info(f"Output Topic:    {KAFKA_TOPIC_TRIGGERS}")
    logger.info(f"Threshold:       {TRIGGER_THRESHOLD}")
    logger.info("=" * 60)

    # Validate environment
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise ValueError("SUPABASE_URL and SUPABASE_KEY must be set")

    # Create Spark session
    spark = SparkSession.builder \
        .appName("INTUNEPipelineTrigger") \
        .config("spark.sql.shuffle.partitions", "4") \
        .config("spark.streaming.stopGracefullyOnShutdown", "true") \
        .config("spark.jars.packages", "org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    logger.info("Spark session created")

    # Read from Kafka
    kafka_df = spark.readStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("subscribe", KAFKA_TOPIC_EVENTS) \
        .option("startingOffsets", "latest") \
        .option("failOnDataLoss", "false") \
        .load()

    # Parse JSON values
    events_df = kafka_df.select(
        from_json(col("value").cast("string"), event_schema).alias("data")
    ).select("data.*")

    # Convert string timestamp to actual Spark TimestampType
    events_df = events_df.withColumn("event_ts", col("event_ts").cast("timestamp"))

    # Add watermark for late data (5-minute tolerance)
    events_df = events_df.withWatermark("event_ts", "5 minutes")

    logger.info("Kafka stream configured")

    # ========================================================================
    # STREAM 1: Aggregate counts by status_eval_first
    # ========================================================================

    # Filter for events where status_eval_first changed to 'done'
    done_events_df = events_df.filter(col("status_eval_first") == TARGET_STATUS)

    # Count by status_eval_first
    counts_df = done_events_df \
        .groupBy("status_eval_first") \
        .count()

    # Write counts to Supabase
    count_query = counts_df.writeStream \
        .foreachBatch(write_counts_to_supabase) \
        .outputMode("complete") \
        .option("checkpointLocation", f"{SPARK_CHECKPOINT_DIR}/counts") \
        .start()

    logger.info("Count upsert stream started")

    # ========================================================================
    # STREAM 2: Trigger emission when threshold reached
    # ========================================================================

    # Create trigger event when count >= threshold
    # Note: This is a simplified approach. Production would need stateful
    # processing to ensure trigger is emitted exactly once.

    trigger_candidates_df = counts_df.filter(col("count") >= TRIGGER_THRESHOLD)

    # Map to trigger event structure
    triggers_df = trigger_candidates_df.selectExpr(
        "uuid() as trigger_id",
        f"'{TRIGGER_STAGE}' as stage",
        "'threshold_reached' as reason",
        f"{TRIGGER_THRESHOLD} as threshold",
        "count as observed_count",
        "current_timestamp() as fired_at",
        f"CONCAT('finetune_', {TRIGGER_THRESHOLD}) as dedupe_key"
    )

    # Convert to JSON and write to Kafka
    trigger_json_df = triggers_df.select(
        col("stage").cast("string").alias("key"),
        to_json(struct("*")).alias("value")
    )

    trigger_query = trigger_json_df.writeStream \
        .format("kafka") \
        .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
        .option("topic", KAFKA_TOPIC_TRIGGERS) \
        .option("checkpointLocation", f"{SPARK_CHECKPOINT_DIR}/triggers") \
        .outputMode("complete") \
        .start()

    logger.info("Trigger emission stream started")

    # ========================================================================
    # AWAIT TERMINATION
    # ========================================================================

    logger.info("Spark streaming job started. Press Ctrl+C to stop.")

    try:
        count_query.awaitTermination()
        trigger_query.awaitTermination()
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
    finally:
        logger.info("Stopping Spark session...")
        spark.stop()
        logger.info("Spark session stopped")


if __name__ == "__main__":
    main()
