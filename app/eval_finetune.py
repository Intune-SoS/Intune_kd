"""
Worker to handle fine-tuning and post-finetune evaluation.

TRIGGER MODE:
    The functions in this file are designed to be called by trigger_consumer.py
    when the threshold of status_eval_first='done' records is reached (trigger event).

    Event-driven workflow (NO POLLING):
      1. prepare_training_data() - Fetch all records with status_eval_first='done'
      2. run_finetune() - Execute finetune.py
      3. evaluate_with_finetuned_model() - Evaluate ALL pending records in one pass

MANUAL MODE (Legacy):
    Can still be run directly for testing: python app/eval_finetune.py
    This will execute the old polling behavior for backward compatibility.

ASYNC POLLING BASELINE (Research measurement):
    main_async_polling_baseline() implements a realistic production polling system
    where data prefetching for checkpoint N+1 runs in a background thread while
    checkpoint N is training.  This is used to produce a fair comparison against
    the event-driven pipeline for Table 6 of the INTUNE paper.

    The transition latency measured here is:
        time_threshold_crossed  →  time_run_finetune_called
    accounting for the fact that prepare_training_data() has already completed
    in the background, so only the remaining poll-sleep time contributes.
"""
import os
import sys
import time
import threading
import subprocess
import logging

# Windows compatibility
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_COMPILE_THREADS"] = "1"

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.database.supabase_client import get_supabase_client
from src.metrics.llm_eval import score_datapoint

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def to_int8(value):
    """Convert decimal metric to int8"""
    if value is None:
        return None
    return int(round(value * 10000))


def check_finetune_conditions():
    """Check if we have 2 records with status_eval_first='done' and empty status_eval_final"""
    try:
        supabase = get_supabase_client()
        
        response = supabase.table("intune_db")\
            .select("id", count="exact")\
            .eq("status_eval_first", "done")\
            .is_("status_eval_final", "null")\
            .execute()
        
        count = response.count or 0
        logger.info(f"Found {count} records ready for fine-tuning")
        return count >= 2
    except Exception as e:
        logger.error(f"Error checking conditions: {e}")
        return False


def prepare_training_data():
    """Fetch training data from Supabase and create JSONL files for fine-tuning"""
    try:
        logger.info("Fetching training data from Supabase...")
        supabase = get_supabase_client()
        
        # Fetch records with status_eval_first='done' and status_eval_final=null
        response = supabase.table("intune_db")\
            .select("*")\
            .eq("status_eval_first", "done")\
            .is_("status_eval_final", "null")\
            .execute()
        
        records = response.data
        logger.info(f"Fetched {len(records)} records for training")
        
        if len(records) < 2:
            logger.warning("Not enough records for training")
            return False
        
        # Create training dataset in the format expected by finetune.py
        train_data = []
        for record in records:
            item = {
                "instruction": "Answer the following question accurately and concisely based on the provided information.",
                "input": record.get("input", ""),
                "output": record.get("expected_output", "") or record.get("actual_output", "")
            }
            train_data.append(item)
        
        # Split into train/val (80/20)
        split_idx = int(len(train_data) * 0.8)
        train_set = train_data[:split_idx]
        val_set = train_data[split_idx:]
        
        # Ensure data directories exist
        data_dir = os.path.join(project_root, 'data', 'processed')
        os.makedirs(data_dir, exist_ok=True)
        
        # Write JSONL files
        import json
        train_file = os.path.join(data_dir, 'train_dataset.jsonl')
        val_file = os.path.join(data_dir, 'val_dataset.jsonl')
        
        with open(train_file, 'w', encoding='utf-8') as f:
            for item in train_set:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        with open(val_file, 'w', encoding='utf-8') as f:
            for item in val_set:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        
        logger.info(f"✓ Created training dataset: {len(train_set)} train, {len(val_set)} val")
        return True
        
    except Exception as e:
        logger.error(f"Error preparing training data: {e}")
        return False


def run_finetune():
    """Execute the real finetune.py script"""
    try:
        finetune_script = os.path.join(project_root, 'src', 'training', 'finetune.py')

        if not os.path.exists(finetune_script):
            logger.error(f"Finetune script not found: {finetune_script}")
            return False

        logger.info("Starting real fine-tuning process...")

        process = subprocess.Popen(
            [sys.executable, finetune_script],
            cwd=project_root,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1
        )
        
        for line in iter(process.stdout.readline, ''):
            if line:
                logger.info(f"Finetune: {line.strip()}")
        
        return_code = process.wait()
        
        if return_code == 0:
            logger.info("✅ Fine-tuning completed successfully")
            return True
        else:
            logger.error(f"❌ Fine-tuning failed with code {return_code}")
            return False
    except Exception as e:
        logger.error(f"Error running finetune: {e}")
        return False


def load_finetuned_model():
    """Load fine-tuned model for inference using transformers + PEFT (Python 3.9 compatible)"""
    try:
        model_path = os.path.join(project_root, 'models', 'gemma-finetuned-merged')

        if not os.path.exists(model_path):
            logger.error(f"Fine-tuned model not found: {model_path}")
            logger.info("Please run finetuning first to create the model")
            return None, None

        # Use transformers instead of unsloth for Python 3.9 compatibility
        from transformers import AutoModelForCausalLM, AutoTokenizer
        import torch

        logger.info(f"Loading fine-tuned model from {model_path}...")

        # Load the merged model (no PEFT needed since it's already merged)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto" if torch.cuda.is_available() else "cpu",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

        # Set to eval mode for inference
        model.eval()

        logger.info(f"✓ Loaded fine-tuned model from {model_path}")
        return model, tokenizer

    except Exception as e:
        logger.error(f"Error loading model: {e}")
        return None, None


def format_context(context):
    """Format context into string"""
    if not context:
        return ""
    if isinstance(context, list):
        return "\n".join(f"- {item}" for item in context if item)
    return str(context)


def generate_with_finetuned(model, tokenizer, record):
    """Generate output using fine-tuned model"""
    try:
        import torch

        question = record.get("input", "")
        context = format_context(record.get("context"))

        instruction = "Answer the following question accurately and concisely based on the provided information."

        if context:
            input_text = f"Context:\n{context}\n\nQuestion: {question}"
        else:
            input_text = f"Question: {question}"

        prompt = f"Human: {instruction}\n\n{input_text}\nAssistant:"

        # Tokenize input
        inputs = tokenizer([prompt], return_tensors="pt")

        # Move to same device as model
        if hasattr(model, 'device'):
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
        elif torch.cuda.is_available():
            inputs = {k: v.to("cuda") for k, v in inputs.items()}

        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
                eos_token_id=tokenizer.eos_token_id,
                use_cache=True
            )

        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Extract just the model's response (after "Assistant:")
        if "Assistant:" in response:
            response = response.split("Assistant:")[-1]
        if "<|endoftext|>" in response:
            response = response.split("<|endoftext|>")[0]

        response = response.strip()

        logger.debug(f"Generated response: {response[:100]}...")
        return response

    except Exception as e:
        logger.error(f"Error generating response: {e}")
        return None


def compute_finetuned_metrics(record, output):
    """Compute metrics for fine-tuned output"""
    try:
        item = {
            "input": record.get("input", ""),
            "expected_output": record.get("expected_output", ""),
            "context": record.get("context", []),
            "actual_output": output
        }
        
        metrics = score_datapoint(item)
        return {k: round(v, 4) if isinstance(v, (int, float)) else v for k, v in metrics.items()}
    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        return None


def update_finetuned_record(record_id, output, metrics):
    """Update record with fine-tuned output and metrics"""
    try:
        supabase = get_supabase_client()
        
        update_data = {
            "actual_output_tuned": output,
            "answer_relevancy_tuned": to_int8(metrics.get("answer_relevancy")),
            "contextual_precision_tuned": to_int8(metrics.get("contextual_precision")),
            "contextual_recall_tuned": to_int8(metrics.get("contextual_recall")),
            "contextual_relevancy_tuned": to_int8(metrics.get("contextual_relevancy")),
            "faithfulness_tuned": to_int8(metrics.get("faithfulness")),
            "toxicity_tuned": to_int8(metrics.get("toxicity")),
            "hallucination_rate_tuned": to_int8(metrics.get("hallucination_rate")),
            "overall_tuned": to_int8(metrics.get("overall")),
            "status_eval_final": "done"
        }
        
        supabase.table("intune_db").update(update_data).eq("id", record_id).execute()
        return True
    except Exception as e:
        logger.error(f"Error updating record {record_id}: {e}")
        return False


def evaluate_with_finetuned_model():
    """
    Evaluate all pending records with fine-tuned model in ONE PASS (no polling).

    TRIGGER MODE: Processes all records with status_eval_first='done'
    and status_eval_final=null in batches until complete.
    """
    logger.info("Starting post-finetune evaluation...")

    model, tokenizer = load_finetuned_model()
    if model is None or tokenizer is None:
        logger.error("Cannot load model for evaluation")
        return False

    try:
        supabase = get_supabase_client()

        total_processed = 0
        batch_size = 5

        while True:
            # Fetch records needing final evaluation
            response = supabase.table("intune_db")\
                .select("*")\
                .eq("status_eval_first", "done")\
                .is_("status_eval_final", "null")\
                .limit(batch_size)\
                .execute()

            records = response.data

            if not records:
                logger.info(f"All records evaluated with fine-tuned model (total: {total_processed})")
                break

            logger.info(f"Evaluating batch of {len(records)} records...")

            for record in records:
                record_id = record.get("id")
                logger.info(f"Processing record {record_id}")

                output = generate_with_finetuned(model, tokenizer, record)

                if output:
                    metrics = compute_finetuned_metrics(record, output)

                    if metrics:
                        if update_finetuned_record(record_id, output, metrics):
                            logger.info(f"✓ Updated record {record_id}")
                            total_processed += 1
                        else:
                            logger.error(f"✗ Failed to update record {record_id}")
                    else:
                        logger.error(f"✗ Failed to compute metrics for record {record_id}")
                else:
                    logger.error(f"✗ Failed to generate output for record {record_id}")

            # Small pause between batches
            time.sleep(2)

        return True
    except Exception as e:
        logger.error(f"Error during evaluation: {e}")
        return False


def main():
    """
    MANUAL MODE: Legacy polling behavior for backward compatibility.

    For production, use trigger_consumer.py which calls these functions
    when Kafka trigger events arrive (event-driven, no polling).
    """
    logger.info("Starting fine-tune worker (MANUAL MODE)...")
    logger.info("⚠️  For event-driven execution, use trigger_consumer.py")

    finetune_done = False

    while True:
        try:
            if not finetune_done:
                # Check if we should run fine-tuning
                if check_finetune_conditions():
                    logger.info("🎯 Conditions met! Preparing training data...")

                    # Prepare training data from Supabase
                    if not prepare_training_data():
                        logger.error("Failed to prepare training data, retrying...")
                        time.sleep(300)
                        continue

                    logger.info("Starting fine-tuning...")

                    if run_finetune():
                        logger.info("✅ Fine-tuning completed")
                        finetune_done = True
                    else:
                        logger.error("❌ Fine-tuning failed, will retry")
                        time.sleep(600)  # Wait 10 minutes before retry
                else:
                    logger.info("⏳ Waiting for 2 evaluated records...")
                    time.sleep(300)  # Check every 5 minutes
            else:
                # Fine-tuning done, now evaluate with fine-tuned model
                logger.info("Starting final evaluation with fine-tuned model...")

                if evaluate_with_finetuned_model():
                    logger.info("🎉 All evaluations complete!")
                    break  # Exit after completing all evaluations
                else:
                    logger.error("Evaluation incomplete, retrying...")
                    time.sleep(60)

        except Exception as e:
            logger.error(f"Error in worker loop: {e}")
            time.sleep(60)

    logger.info("Worker finished")


# =============================================================================
# ASYNC POLLING BASELINE — Research measurement for INTUNE paper Table 6
# =============================================================================
#
# Design rationale
# ----------------
# The naive polling baseline (main() above) is unfair to compare against the
# event-driven system because it assumes the system does nothing while training
# is running.  A real production system would overlap work: while checkpoint N
# trains, a background thread watches the DB and prefetches data for checkpoint
# N+1.  When the next poll fires and the threshold is met, data is already on
# disk — the only remaining latency is the tail of the current poll sleep.
#
# What we measure
# ---------------
# Transition latency = time from when check_finetune_conditions() first returns
# True  →  time when run_finetune() (or its mock) is actually called.
#
# In the async baseline this is:
#   - If prefetch already finished: ~0 s  (data ready, training starts immediately)
#   - If prefetch still running:    time to wait for prefetch thread to join
#   - Worst case (no prefetch yet): full prepare_training_data() duration
#
# The poll interval (300 s) determines *when* the condition is first noticed,
# but that is separate from the transition latency once it is noticed.  The
# measurement script (measure_latency.py) controls exactly when the threshold
# is crossed relative to the poll cycle so we can isolate the prefetch benefit.
#
# run_finetune() is replaced by mock_run_finetune() (time.sleep(120)) so the
# measurement runs in ~2 minutes instead of hours.
# =============================================================================

POLL_INTERVAL_SECONDS = 300  # unchanged — do not modify


class _PrefetchState:
    """
    Shared state between the main polling loop and the background prefetch thread.

    Attributes
    ----------
    thread : threading.Thread | None
        The currently running prefetch thread, or None if no prefetch is active.
    ready : bool
        True once prepare_training_data() has completed successfully in the
        background and the JSONL files are on disk.
    lock : threading.Lock
        Protects reads/writes to `ready` and `thread`.
    """

    def __init__(self):
        self.thread: threading.Thread | None = None
        self.ready: bool = False
        self.lock: threading.Lock = threading.Lock()

    def reset(self):
        """Reset state for the next checkpoint cycle."""
        with self.lock:
            self.thread = None
            self.ready = False


def _prefetch_worker(state: _PrefetchState) -> None:
    """
    Background thread target: calls prepare_training_data() and sets state.ready.

    This runs while the current checkpoint is training so that data for the
    next checkpoint is already on disk when the poll condition fires.
    """
    logger.info("[ASYNC-POLL] Background prefetch started")
    try:
        success = prepare_training_data()
        with state.lock:
            state.ready = success
        if success:
            logger.info("[ASYNC-POLL] Background prefetch complete — data ready on disk")
        else:
            logger.warning("[ASYNC-POLL] Background prefetch returned False — data may be incomplete")
    except Exception as exc:
        logger.error(f"[ASYNC-POLL] Background prefetch raised: {exc}")
        with state.lock:
            state.ready = False


def mock_run_finetune() -> bool:
    """
    Mock fine-tuning for latency measurement only.

    Sleeps for 120 seconds to simulate a 2-minute training step without
    consuming GPU memory or touching model weights.  Replace with run_finetune()
    for production use.
    """
    logger.info("[ASYNC-POLL] mock_run_finetune: simulating 2-minute training step")
    time.sleep(120)
    logger.info("[ASYNC-POLL] mock_run_finetune: done")
    return True


def main_async_polling_baseline(
    num_transitions: int = 3,
    use_mock_finetune: bool = True,
) -> list[float]:
    """
    Async polling baseline for INTUNE paper Table 6.

    Runs `num_transitions` checkpoint transitions and returns a list of
    transition latencies (seconds).  Each latency is the time from when
    check_finetune_conditions() first returns True to when training starts,
    accounting for background prefetching.

    Parameters
    ----------
    num_transitions : int
        Number of checkpoint transitions to measure (paper uses 3).
    use_mock_finetune : bool
        If True, replace run_finetune() with mock_run_finetune() (120 s sleep).
        Set to False only when running a real training experiment.

    Returns
    -------
    list[float]
        Transition latency in seconds for each measured transition.
    """
    logger.info("=" * 60)
    logger.info("ASYNC POLLING BASELINE — INTUNE Table 6 measurement")
    logger.info(f"  Poll interval : {POLL_INTERVAL_SECONDS} s")
    logger.info(f"  Transitions   : {num_transitions}")
    logger.info(f"  Mock finetune : {use_mock_finetune}")
    logger.info("=" * 60)

    finetune_fn = mock_run_finetune if use_mock_finetune else run_finetune
    transition_latencies: list[float] = []
    state = _PrefetchState()

    for transition_idx in range(1, num_transitions + 1):
        logger.info(f"\n--- Transition {transition_idx}/{num_transitions} ---")
        state.reset()

        # ------------------------------------------------------------------ #
        # PHASE 1: Poll until threshold is met                                #
        # ------------------------------------------------------------------ #
        threshold_crossed_at: float | None = None

        while threshold_crossed_at is None:
            logger.info(f"[ASYNC-POLL] Polling DB (interval={POLL_INTERVAL_SECONDS}s)...")
            time.sleep(POLL_INTERVAL_SECONDS)

            if check_finetune_conditions():
                threshold_crossed_at = time.time()
                logger.info(
                    f"[ASYNC-POLL] Threshold met at t={threshold_crossed_at:.3f}"
                )
            else:
                logger.info("[ASYNC-POLL] Threshold not yet met, sleeping again")

                # While waiting, kick off a speculative prefetch if one is not
                # already running.  This mirrors what a real production system
                # would do: start preparing data as soon as there is *any*
                # activity, even before the threshold is confirmed.
                with state.lock:
                    prefetch_running = state.thread is not None and state.thread.is_alive()

                if not prefetch_running:
                    logger.info("[ASYNC-POLL] Launching speculative prefetch thread")
                    t = threading.Thread(
                        target=_prefetch_worker,
                        args=(state,),
                        daemon=True,
                        name=f"prefetch-transition-{transition_idx}",
                    )
                    with state.lock:
                        state.thread = t
                    t.start()

        # ------------------------------------------------------------------ #
        # PHASE 2: Threshold crossed — wait for prefetch if still running,   #
        #          then start training.  Measure the gap.                     #
        # ------------------------------------------------------------------ #

        # Check whether prefetch data is already ready
        with state.lock:
            data_ready = state.ready
            prefetch_thread = state.thread

        if data_ready:
            logger.info("[ASYNC-POLL] Data already prefetched — starting training immediately")
        elif prefetch_thread is not None and prefetch_thread.is_alive():
            logger.info("[ASYNC-POLL] Prefetch still running — waiting for it to finish")
            prefetch_thread.join()
            with state.lock:
                data_ready = state.ready
            if data_ready:
                logger.info("[ASYNC-POLL] Prefetch finished — data ready")
            else:
                logger.warning("[ASYNC-POLL] Prefetch finished but data not ready — running prepare now")
                prepare_training_data()
        else:
            # No prefetch was started (threshold crossed on the very first poll
            # before any speculative prefetch could be launched).
            logger.info("[ASYNC-POLL] No prefetch available — running prepare_training_data() now")
            prepare_training_data()

        # Record the moment training is about to start
        training_start_at = time.time()
        transition_latency = training_start_at - threshold_crossed_at

        print(
            f"\n[METRIC ASYNC-POLL] TRANSITION LATENCY: {transition_latency:.3f}s"
            f"  (transition {transition_idx}/{num_transitions})"
        )
        logger.info(
            f"[ASYNC-POLL] Transition latency = {transition_latency:.3f}s"
        )

        transition_latencies.append(transition_latency)

        # ------------------------------------------------------------------ #
        # PHASE 3: Run (mock) fine-tuning                                     #
        # ------------------------------------------------------------------ #
        logger.info(f"[ASYNC-POLL] Starting finetune for transition {transition_idx}")
        finetune_fn()

        # While this checkpoint trains, launch the prefetch for the next one
        # so it is ready when the next poll cycle fires.
        if transition_idx < num_transitions:
            logger.info("[ASYNC-POLL] Launching prefetch for next checkpoint while training")
            state.reset()
            t = threading.Thread(
                target=_prefetch_worker,
                args=(state,),
                daemon=True,
                name=f"prefetch-transition-{transition_idx + 1}",
            )
            with state.lock:
                state.thread = t
            t.start()

        logger.info(f"[ASYNC-POLL] Transition {transition_idx} complete")

    # Summary
    mean_latency = sum(transition_latencies) / len(transition_latencies)
    logger.info("=" * 60)
    logger.info("ASYNC POLLING BASELINE — RESULTS")
    for i, lat in enumerate(transition_latencies, 1):
        logger.info(f"  Transition {i}: {lat:.3f}s")
    logger.info(f"  Mean latency : {mean_latency:.3f}s")
    logger.info("=" * 60)

    return transition_latencies


if __name__ == "__main__":
    main()
