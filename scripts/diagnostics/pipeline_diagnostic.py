#!/usr/bin/env python3
"""
Pipeline Diagnostic Tool - Find where the pipeline is failing
============================================================
"""
import os
import sys
import asyncio
import json
import logging
from dotenv import load_dotenv
from confluent_kafka import Consumer, Producer

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from supabase import acreate_client
from src.database.supabase_client import get_supabase_client

load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
KAFKA_BOOTSTRAP_SERVERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092")

async def test_1_check_data_in_db():
    """Test 1: Check if test data exists in database"""
    logger.info("🔍 TEST 1: Checking test data in database...")

    try:
        supabase = get_supabase_client()

        # Check records with status_eval_first='done'
        response = supabase.table("intune_db")\
            .select("id,input,status_eval_first,status_eval_final")\
            .eq("status_eval_first", "done")\
            .order("id", desc=True)\
            .limit(5)\
            .execute()

        records = response.data
        logger.info(f"✅ Found {len(records)} records with status_eval_first='done'")

        for record in records:
            logger.info(f"   ID: {record['id']}, status_first: {record['status_eval_first']}, status_final: {record['status_eval_final']}")

        return len(records) > 0

    except Exception as e:
        logger.error(f"❌ Database test failed: {e}")
        return False

async def test_2_realtime_subscription():
    """Test 2: Test if Realtime can detect changes"""
    logger.info("🔍 TEST 2: Testing Realtime subscription...")

    changes_detected = []

    try:
        supabase = await acreate_client(SUPABASE_URL, SUPABASE_KEY)

        def handle_change(payload):
            logger.info(f"🎯 REALTIME CHANGE DETECTED!")
            logger.info(f"   Event: {payload.get('eventType')}")
            logger.info(f"   Table: {payload.get('table')}")

            record = payload.get('new', {}) or payload.get('record', {})
            logger.info(f"   Record ID: {record.get('id')}")
            logger.info(f"   Status First: {record.get('status_eval_first')}")
            logger.info(f"   Status Final: {record.get('status_eval_final')}")

            changes_detected.append(payload)

        # Subscribe to intune_db changes
        channel = supabase.channel('diagnostic-test')
        await channel.on_postgres_changes(
            event='*',
            schema='public',
            table='intune_db',
            callback=handle_change
        ).subscribe()

        logger.info("✅ Subscribed to intune_db changes")
        logger.info("💡 Now manually update a record (or run manual_test_data_generator.py option 5)")
        logger.info("⏳ Listening for 30 seconds...")

        # Listen for 30 seconds
        await asyncio.sleep(30)

        await channel.unsubscribe()
        await supabase.aclose()

        logger.info(f"✅ Test complete. Detected {len(changes_detected)} changes")
        return len(changes_detected) > 0

    except Exception as e:
        logger.error(f"❌ Realtime test failed: {e}")
        return False

def test_3_kafka_topics():
    """Test 3: Check if Kafka topics exist and can be used"""
    logger.info("🔍 TEST 3: Testing Kafka topics...")

    try:
        # Test producer
        producer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'enable.idempotence': True,
            'acks': 'all'
        }
        producer = Producer(producer_config)

        # Test message
        test_event = {
            "test": "diagnostic",
            "timestamp": "2026-03-20T12:00:00Z",
            "status_eval_first": "done"
        }

        topic = "intune.status.events"
        logger.info(f"📤 Publishing test message to {topic}...")

        producer.produce(
            topic=topic,
            key="diagnostic",
            value=json.dumps(test_event).encode('utf-8')
        )
        producer.flush(timeout=10)
        logger.info("✅ Message published successfully")

        # Test consumer
        consumer_config = {
            'bootstrap.servers': KAFKA_BOOTSTRAP_SERVERS,
            'group.id': 'diagnostic-consumer',
            'auto.offset.reset': 'latest'
        }
        consumer = Consumer(consumer_config)
        consumer.subscribe([topic])

        logger.info(f"📥 Consuming from {topic} (5 second timeout)...")
        messages_received = 0

        for _ in range(5):  # 5 second timeout
            msg = consumer.poll(timeout=1.0)
            if msg is None:
                continue
            if msg.error():
                logger.error(f"Consumer error: {msg.error()}")
                continue

            try:
                value = json.loads(msg.value().decode('utf-8'))
                if value.get("test") == "diagnostic":
                    logger.info("✅ Received our test message!")
                    messages_received += 1
                else:
                    logger.info(f"📨 Received message: {value}")
                    messages_received += 1
            except Exception as e:
                logger.error(f"Error parsing message: {e}")

        consumer.close()
        logger.info(f"✅ Kafka test complete. Received {messages_received} messages")
        return True

    except Exception as e:
        logger.error(f"❌ Kafka test failed: {e}")
        return False

def test_4_check_bridge_logs():
    """Test 4: Instructions for checking bridge logs"""
    logger.info("🔍 TEST 4: Bridge Log Analysis")
    logger.info("📋 Check your realtime_kafka_bridge.py terminal for:")
    logger.info("   ✅ 'Bridge started. Listening on Supabase Realtime.'")
    logger.info("   ✅ 'Published event: record_id=XXXX status_eval_first=done'")
    logger.info("   ❌ 'Skipping event with missing status fields'")
    logger.info("   ❌ 'Failed to publish event'")
    logger.info("   ❌ 'Realtime connection error'")

async def main():
    """Run all diagnostic tests"""
    logger.info("🚀 Pipeline Diagnostic Tool")
    logger.info("=" * 50)

    # Test 1: Data in database
    test1_result = await test_1_check_data_in_db()

    # Test 2: Realtime subscription
    test2_result = await test_2_realtime_subscription()

    # Test 3: Kafka topics
    test3_result = test_3_kafka_topics()

    # Test 4: Bridge logs instruction
    test_4_check_bridge_logs()

    logger.info("=" * 50)
    logger.info("📊 DIAGNOSTIC RESULTS:")
    logger.info(f"   Database:  {'✅' if test1_result else '❌'}")
    logger.info(f"   Realtime:  {'✅' if test2_result else '❌'}")
    logger.info(f"   Kafka:     {'✅' if test3_result else '❌'}")

    if not test1_result:
        logger.info("🔧 Fix: Run manual_test_data_generator.py option 5 to insert test data")

    if not test2_result:
        logger.info("🔧 Fix: Check Supabase Dashboard → Database → Tables → intune_db → Enable Realtime")

    if not test3_result:
        logger.info("🔧 Fix: Start Kafka using: docker-compose -f docker-compose-kafka.yml up -d")

if __name__ == "__main__":
    asyncio.run(main())