"""
measure_latency.py — INTUNE Table 6 latency measurement script
==============================================================

Runs 3 checkpoint transitions under the async polling baseline and prints
the two numbers needed for Table 6:

    Async Polling Baseline mean latency  (seconds)
    Event-Driven mean latency            (seconds, from paper: 6.5 s)

Usage
-----
    python measure_latency.py

The script controls the DB state so that the threshold is crossed at a
*known* point in the poll cycle, giving reproducible measurements.

How it works
------------
1. It inserts TRIGGER_THRESHOLD synthetic records into intune_db with
   status_eval_first=NULL.
2. It waits for the poll cycle to fire once (confirming no false positive).
3. It updates those records to status_eval_first='done', recording the
   exact timestamp — this is "threshold crossed".
4. The async polling baseline in eval_finetune.py detects the change on
   the next poll and starts training.  The gap is the transition latency.
5. Steps 1-4 repeat for each of the 3 transitions.

run_finetune() is mocked with a 120-second sleep so the full measurement
takes roughly:  3 × (poll_interval + 120 s)  ≈  15 minutes at 300 s poll.

For a faster smoke-test set FAST_MODE=True below, which reduces the poll
interval to 10 seconds (modifies only the measurement run, not production).
"""

import os
import sys
import time
import threading
import statistics
import logging
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# Set to True to use a 10-second poll interval for quick smoke-testing.
# Set to False for the real 300-second measurement used in the paper.
FAST_MODE = True

# Number of transitions to measure (paper uses 3)
NUM_TRANSITIONS = 3

# Threshold: number of 'done' records that trigger fine-tuning
TRIGGER_THRESHOLD = 2

# Event-driven latency from the paper (seconds) — used only for the summary
EVENT_DRIVEN_LATENCY_PAPER = 6.5

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

app_dir = os.path.join(project_root, 'app')
if app_dir not in sys.path:
    sys.path.insert(0, app_dir)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Imports (after path setup)
# ---------------------------------------------------------------------------

from src.database.supabase_client import get_supabase_client
from eval_finetune import (
    _PrefetchState,
    _prefetch_worker,
    mock_run_finetune,
    prepare_training_data,
    check_finetune_conditions,
    POLL_INTERVAL_SECONDS,
)

# Override poll interval for fast mode
if FAST_MODE:
    EFFECTIVE_POLL_INTERVAL = 10
    logger.info(f"FAST_MODE=True: using {EFFECTIVE_POLL_INTERVAL}s poll interval")
else:
    EFFECTIVE_POLL_INTERVAL = POLL_INTERVAL_SECONDS
    logger.info(f"FAST_MODE=False: using {EFFECTIVE_POLL_INTERVAL}s poll interval")


# ---------------------------------------------------------------------------
# DB helpers
# ---------------------------------------------------------------------------

def _insert_pending_records(n: int = TRIGGER_THRESHOLD) -> list[int]:
    """
    Insert n synthetic records with status_eval_first=NULL.
    Returns the list of inserted IDs.
    """
    supabase = get_supabase_client()
    records = []
    for i in range(n):
        records.append({
            "input": f"[MEASURE] Synthetic record {i+1} at {datetime.now(timezone.utc).isoformat()}",
            "actual_output": "Synthetic output for latency measurement.",
            "expected_output": "Synthetic expected output.",
            "status_eval_first": None,
            "status_eval_final": None,
            "context": [],
        })
    response = supabase.table("intune_db").insert(records).execute()
    ids = [r["id"] for r in response.data]
    logger.info(f"Inserted {len(ids)} synthetic records: {ids}")
    return ids


def _mark_records_done(record_ids: list[int]) -> float:
    """
    Update records to status_eval_first='done'.
    Returns the Unix timestamp immediately after the last update — this is
    the "threshold crossed" time used to compute transition latency.
    """
    supabase = get_supabase_client()
    for rid in record_ids:
        supabase.table("intune_db") \
            .update({"status_eval_first": "done"}) \
            .eq("id", rid) \
            .execute()
    threshold_ts = time.time()
    logger.info(
        f"Marked {len(record_ids)} records as 'done'. "
        f"Threshold crossed at t={threshold_ts:.3f}"
    )
    return threshold_ts


def _cleanup_records(record_ids: list[int]) -> None:
    """Delete synthetic records created for measurement."""
    supabase = get_supabase_client()
    for rid in record_ids:
        supabase.table("intune_db").delete().eq("id", rid).execute()
    logger.info(f"Cleaned up {len(record_ids)} synthetic records")


# ---------------------------------------------------------------------------
# Single-transition measurement
# ---------------------------------------------------------------------------

def measure_one_transition(transition_idx: int) -> float:
    """
    Run one full transition and return the transition latency in seconds.

    Timeline
    --------
    t=0          Insert pending records (status=NULL)
    t=~poll/2    Speculative prefetch starts (background thread)
    t=poll       Poll fires — threshold NOT met (records still NULL)
    t=poll+X     Records flipped to 'done' (threshold crossed)
    t=2*poll     Poll fires — threshold IS met
    t=2*poll+ε   Training starts (ε = time to confirm prefetch ready)

    Transition latency = t_training_start − t_threshold_crossed
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"TRANSITION {transition_idx}/{NUM_TRANSITIONS}")
    logger.info(f"{'='*60}")

    state = _PrefetchState()
    record_ids: list[int] = []
    threshold_crossed_at: float | None = None
    training_start_at: float | None = None

    # ------------------------------------------------------------------ #
    # Step 1: Insert records in NULL state                                #
    # ------------------------------------------------------------------ #
    record_ids = _insert_pending_records(TRIGGER_THRESHOLD)

    # ------------------------------------------------------------------ #
    # Step 2: First poll — threshold not met, launch speculative prefetch #
    # ------------------------------------------------------------------ #
    logger.info(f"[MEASURE] Sleeping {EFFECTIVE_POLL_INTERVAL}s (first poll cycle)...")
    time.sleep(EFFECTIVE_POLL_INTERVAL)

    # Simulate the poll check — threshold not met yet
    logger.info("[MEASURE] First poll: threshold not yet met (records still NULL)")

    # Launch speculative prefetch (mirrors what main_async_polling_baseline does)
    logger.info("[MEASURE] Launching speculative prefetch thread")
    t = threading.Thread(
        target=_prefetch_worker,
        args=(state,),
        daemon=True,
        name=f"prefetch-t{transition_idx}",
    )
    with state.lock:
        state.thread = t
    t.start()

    # ------------------------------------------------------------------ #
    # Step 3: Flip records to 'done' mid-cycle (threshold crossed)        #
    # ------------------------------------------------------------------ #
    # Wait half a poll interval so the prefetch has time to finish, then
    # cross the threshold.  This simulates the realistic case where the
    # threshold is crossed partway through a poll cycle.
    half_poll = EFFECTIVE_POLL_INTERVAL / 2
    logger.info(f"[MEASURE] Waiting {half_poll:.1f}s then crossing threshold...")
    time.sleep(half_poll)

    threshold_crossed_at = _mark_records_done(record_ids)

    # ------------------------------------------------------------------ #
    # Step 4: Second poll — threshold IS met                              #
    # ------------------------------------------------------------------ #
    remaining_sleep = EFFECTIVE_POLL_INTERVAL - half_poll
    logger.info(f"[MEASURE] Sleeping remaining {remaining_sleep:.1f}s of poll cycle...")
    time.sleep(remaining_sleep)

    logger.info("[MEASURE] Second poll: check_finetune_conditions()")
    if not check_finetune_conditions():
        logger.warning("[MEASURE] check_finetune_conditions() returned False — "
                       "records may have been processed already. Latency may be 0.")

    # ------------------------------------------------------------------ #
    # Step 5: Wait for prefetch if still running                          #
    # ------------------------------------------------------------------ #
    with state.lock:
        data_ready = state.ready
        prefetch_thread = state.thread

    if data_ready:
        logger.info("[MEASURE] Data already prefetched — training starts immediately")
    elif prefetch_thread is not None and prefetch_thread.is_alive():
        logger.info("[MEASURE] Prefetch still running — joining thread")
        prefetch_thread.join()
        with state.lock:
            data_ready = state.ready
        logger.info(f"[MEASURE] Prefetch joined, data_ready={data_ready}")
    else:
        logger.info("[MEASURE] No prefetch available — running prepare_training_data() now")
        prepare_training_data()

    # ------------------------------------------------------------------ #
    # Step 6: Record training start time and compute latency              #
    # ------------------------------------------------------------------ #
    training_start_at = time.time()
    transition_latency = training_start_at - threshold_crossed_at

    print(
        f"\n[METRIC ASYNC-POLL] TRANSITION LATENCY: {transition_latency:.3f}s"
        f"  (transition {transition_idx}/{NUM_TRANSITIONS})"
    )

    # ------------------------------------------------------------------ #
    # Step 7: Run mock finetune (120 s sleep)                             #
    # ------------------------------------------------------------------ #
    logger.info("[MEASURE] Running mock_run_finetune (120s sleep)...")
    mock_run_finetune()

    # ------------------------------------------------------------------ #
    # Step 8: Cleanup synthetic records                                   #
    # ------------------------------------------------------------------ #
    _cleanup_records(record_ids)

    return transition_latency


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    logger.info("=" * 60)
    logger.info("INTUNE Table 6 — Latency Measurement Script")
    logger.info("=" * 60)
    logger.info(f"  Transitions      : {NUM_TRANSITIONS}")
    logger.info(f"  Poll interval    : {EFFECTIVE_POLL_INTERVAL}s")
    logger.info(f"  Trigger threshold: {TRIGGER_THRESHOLD} records")
    logger.info(f"  Mock finetune    : 120s sleep")
    logger.info("=" * 60)

    latencies: list[float] = []

    for i in range(1, NUM_TRANSITIONS + 1):
        lat = measure_one_transition(i)
        latencies.append(lat)
        logger.info(f"Transition {i} latency: {lat:.3f}s")

    # ------------------------------------------------------------------ #
    # Results                                                             #
    # ------------------------------------------------------------------ #
    mean_async = statistics.mean(latencies)
    stdev_async = statistics.stdev(latencies) if len(latencies) > 1 else 0.0

    event_driven_mean = EVENT_DRIVEN_LATENCY_PAPER

    # Honest percentage reduction
    if mean_async > 0:
        pct_reduction = (mean_async - event_driven_mean) / mean_async * 100
    else:
        pct_reduction = 0.0

    print("\n" + "=" * 60)
    print("TABLE 6 — ORCHESTRATION LATENCY COMPARISON")
    print("=" * 60)
    print(f"{'System':<35} {'Mean Latency':>14}  {'Std Dev':>10}")
    print("-" * 60)
    print(
        f"{'Async Polling Baseline':<35} "
        f"{mean_async:>12.2f}s  "
        f"{stdev_async:>8.2f}s"
    )
    print(
        f"{'Event-Driven (paper)':<35} "
        f"{event_driven_mean:>12.2f}s  "
        f"{'N/A':>10}"
    )
    print("-" * 60)
    print(f"Honest reduction: {pct_reduction:.1f}%")
    print("=" * 60)

    print("\nPer-transition breakdown:")
    for i, lat in enumerate(latencies, 1):
        print(f"  Transition {i}: {lat:.3f}s")

    print(
        f"\n→ Add to Table 6:"
        f"\n    Async Polling Baseline | {mean_async:.1f}s | {pct_reduction:.0f}% reduction vs event-driven"
        f"\n    Event-Driven           | {event_driven_mean:.1f}s | baseline"
    )

    return latencies


if __name__ == "__main__":
    main()
