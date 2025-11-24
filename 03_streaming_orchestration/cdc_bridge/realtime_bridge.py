"""
realtime_bridge.py — Compatibility alias forwarding to realtime_kafka_bridge.py
"""
from .realtime_kafka_bridge import main

if __name__ == "__main__":
    main()
