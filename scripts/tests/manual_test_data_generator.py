"""
Manual Test Data Generator for INTUNE Pipeline
==============================================

This script generates random test data for the intune_db table to test
the complete event-driven pipeline:

Pipeline Flow Test:
    1. Insert data with status_eval_first=NULL
    2. Update status_eval_first='done'  ← This triggers the pipeline!
    3. Watch the pipeline process the data automatically

Usage:
    python3 manual_test_data_generator.py
"""

import os
import sys
import logging
import random
import json
import time
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import get_supabase_client

# Load environment variables
load_dotenv()

# Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ============================================================================
# SAMPLE DATA TEMPLATES
# ============================================================================

# Sample questions and expected answers for testing
TEST_QUESTIONS = [
    {
        "input": "What is the capital of France?",
        "expected_output": "The capital of France is Paris."
    },
    {
        "input": "Explain photosynthesis in simple terms.",
        "expected_output": "Photosynthesis is the process by which plants use sunlight, water, and carbon dioxide to create glucose and release oxygen."
    },
    {
        "input": "What is 25 * 4?",
        "expected_output": "25 * 4 = 100"
    },
    {
        "input": "Name three programming languages.",
        "expected_output": "Three popular programming languages are Python, JavaScript, and Java."
    },
    {
        "input": "What causes rain?",
        "expected_output": "Rain is caused when water vapor in clouds condenses into droplets heavy enough to fall to the ground."
    },
    {
        "input": "How do you make tea?",
        "expected_output": "To make tea: boil water, add tea leaves or tea bag to cup, pour hot water over tea, steep for 3-5 minutes, then remove tea bag or strain leaves."
    },
    {
        "input": "What is the largest planet in our solar system?",
        "expected_output": "Jupiter is the largest planet in our solar system."
    },
    {
        "input": "Explain what HTTP stands for.",
        "expected_output": "HTTP stands for HyperText Transfer Protocol, which is used for transferring data over the web."
    },
    {
        "input": "What is the difference between a list and a tuple in Python?",
        "expected_output": "Lists are mutable (can be changed) and use square brackets [], while tuples are immutable (cannot be changed) and use parentheses ()."
    },
    {
        "input": "How many continents are there?",
        "expected_output": "There are 7 continents: Africa, Antarctica, Asia, Europe, North America, Oceania, and South America."
    }
]

# ============================================================================
# DATA GENERATION FUNCTIONS
# ============================================================================

def generate_random_test_data(num_records: int = 5) -> List[Dict]:
    """
    Generate random test records for intune_db table.

    Args:
        num_records: Number of records to generate

    Returns:
        List of dictionaries ready for database insertion
    """
    records = []

    for i in range(num_records):
        # Pick a random question-answer pair
        qa_pair = random.choice(TEST_QUESTIONS)

        # Add timestamp to make each record unique
        timestamp = datetime.now().strftime('%H:%M:%S')
        variation_suffix = f" (Manual Test #{i+1}: {timestamp})"

        record = {
            "input": qa_pair["input"] + variation_suffix,
            "expected_output": qa_pair["expected_output"],
            "actual_output": qa_pair["expected_output"],  # Use expected as initial actual
            "status_eval_first": None,  # Start as NULL
            "status_eval_final": None,  # Start as NULL
            "context": []  # Empty context array
        }

        records.append(record)

    return records

def insert_test_records(records: List[Dict]) -> List[int]:
    """
    Insert test records into intune_db table.

    Args:
        records: List of record dictionaries

    Returns:
        List of inserted record IDs
    """
    try:
        supabase = get_supabase_client()

        logger.info(f"Inserting {len(records)} test records into intune_db...")

        response = supabase.table("intune_db").insert(records).execute()

        inserted_ids = [record["id"] for record in response.data]

        logger.info(f"✅ Successfully inserted {len(inserted_ids)} records")
        logger.info(f"   Record IDs: {inserted_ids}")

        return inserted_ids

    except Exception as e:
        logger.error(f"❌ Error inserting records: {e}")
        return []

def update_status_to_trigger_pipeline(record_ids: List[int], target_count: int = 2):
    """
    Update status_eval_first to 'done' to trigger the pipeline.

    Args:
        record_ids: List of record IDs to update
        target_count: Number of records to update (default 2 for demo threshold)
    """
    try:
        supabase = get_supabase_client()

        # Select which records to update (up to target_count)
        ids_to_update = record_ids[:target_count]

        logger.info(f"🚀 Triggering pipeline: updating {len(ids_to_update)} records to status_eval_first='done'")
        logger.info(f"   Updating record IDs: {ids_to_update}")

        for record_id in ids_to_update:
            response = supabase.table("intune_db")\
                .update({"status_eval_first": "done"})\
                .eq("id", record_id)\
                .execute()

            logger.info(f"   ✓ Updated record ID {record_id}")

        logger.info(f"🎯 Pipeline should trigger! Check your running components:")
        logger.info("   - realtime_kafka_bridge.py (should detect changes)")
        logger.info("   - spark_pipeline_trigger_job_standalone.py (should count events)")
        logger.info("   - trigger_consumer.py (should execute finetune workflow)")

    except Exception as e:
        logger.error(f"❌ Error updating records: {e}")

def check_current_pipeline_status():
    """Check current status of records in the pipeline."""
    try:
        supabase = get_supabase_client()

        logger.info("📊 Current Pipeline Status:")

        # Count by status_eval_first
        for status in [None, "done"]:
            if status is None:
                response = supabase.table("intune_db")\
                    .select("id", count="exact")\
                    .is_("status_eval_first", "null")\
                    .execute()
                status_label = "NULL (not started)"
            else:
                response = supabase.table("intune_db")\
                    .select("id", count="exact")\
                    .eq("status_eval_first", status)\
                    .execute()
                status_label = status

            count = response.count or 0
            logger.info(f"   status_eval_first='{status_label}': {count} records")

        # Count by status_eval_final
        for status in [None, "done"]:
            if status is None:
                response = supabase.table("intune_db")\
                    .select("id", count="exact")\
                    .is_("status_eval_final", "null")\
                    .execute()
                status_label = "NULL (not processed)"
            else:
                response = supabase.table("intune_db")\
                    .select("id", count="exact")\
                    .eq("status_eval_final", status)\
                    .execute()
                status_label = status

            count = response.count or 0
            logger.info(f"   status_eval_final='{status_label}': {count} records")

    except Exception as e:
        logger.error(f"❌ Error checking status: {e}")

# ============================================================================
# INTERACTIVE MENU
# ============================================================================

def show_menu():
    """Show interactive menu for testing."""
    print("\n" + "="*60)
    print("🧪 INTUNE Pipeline Manual Test Data Generator")
    print("="*60)
    print("1. Check current pipeline status")
    print("2. Generate and insert test records (5 records)")
    print("3. Generate and insert custom number of records")
    print("4. Trigger pipeline (update existing records to 'done')")
    print("5. Complete test cycle (insert + trigger)")
    print("6. Exit")
    print("="*60)

def main():
    """Main interactive function."""
    logger.info("🚀 INTUNE Pipeline Test Data Generator Started")

    while True:
        show_menu()

        try:
            choice = input("\nSelect option (1-6): ").strip()

            if choice == "1":
                check_current_pipeline_status()

            elif choice == "2":
                records = generate_random_test_data(5)
                inserted_ids = insert_test_records(records)
                if inserted_ids:
                    logger.info(f"💡 Next step: choose option 4 to trigger pipeline with these records")

            elif choice == "3":
                try:
                    num_records = int(input("Enter number of records to generate: "))
                    if num_records <= 0 or num_records > 50:
                        print("❌ Please enter a number between 1 and 50")
                        continue

                    records = generate_random_test_data(num_records)
                    inserted_ids = insert_test_records(records)
                    if inserted_ids:
                        logger.info(f"💡 Next step: choose option 4 to trigger pipeline with these records")

                except ValueError:
                    print("❌ Please enter a valid number")

            elif choice == "4":
                # Get recent records with status_eval_first=NULL
                try:
                    supabase = get_supabase_client()
                    response = supabase.table("intune_db")\
                        .select("id")\
                        .is_("status_eval_first", "null")\
                        .order("id", desc=True)\
                        .limit(10)\
                        .execute()

                    available_records = [r["id"] for r in response.data]

                    if len(available_records) < 2:
                        logger.warning(f"❌ Need at least 2 records with status_eval_first=NULL, found {len(available_records)}")
                        logger.info("💡 Use option 2 to generate more test records first")
                        continue

                    logger.info(f"Found {len(available_records)} available records: {available_records}")

                    try:
                        target_count = int(input(f"How many records to trigger? (2-{len(available_records)}): "))
                        if target_count < 2 or target_count > len(available_records):
                            print(f"❌ Please enter a number between 2 and {len(available_records)}")
                            continue
                    except ValueError:
                        target_count = 2
                        logger.info("Using default: 2 records")

                    update_status_to_trigger_pipeline(available_records, target_count)

                except Exception as e:
                    logger.error(f"❌ Error: {e}")

            elif choice == "5":
                # Complete test cycle
                logger.info("🔄 Running complete test cycle...")

                # Step 1: Generate records
                records = generate_random_test_data(5)
                inserted_ids = insert_test_records(records)

                if not inserted_ids:
                    logger.error("❌ Failed to insert records, aborting")
                    continue

                # Wait a moment
                logger.info("⏳ Waiting 3 seconds...")
                time.sleep(3)

                # Step 2: Trigger pipeline
                update_status_to_trigger_pipeline(inserted_ids, 2)

                logger.info("✅ Complete test cycle finished!")
                logger.info("📋 Check your pipeline components to see the events flowing through")

            elif choice == "6":
                logger.info("👋 Goodbye!")
                break

            else:
                print("❌ Invalid choice, please select 1-6")

        except KeyboardInterrupt:
            logger.info("\n👋 Goodbye!")
            break
        except Exception as e:
            logger.error(f"❌ Unexpected error: {e}")

if __name__ == "__main__":
    main()