"""
forgetting_analysis.py — Catastrophic forgetting check for INTUNE Table 5.
=========================================================================

Loads ONLY the C4 LoRA adapter and runs inference on the *exact same 100
records* that the C1 adapter was originally evaluated on (from
modelcomp_50k checkpoint=1 — the 100 highest ids, persisted in
reports/incremental/checkpoint_eval_supabase_raw.json).

Compares C4-on-C1's-data against C1-on-C1's-data to surface per-metric
deltas — a negative delta means C4 has forgotten what C1 learned about
that data slice.

Reuses model-loading, prompt-formatting, batched-inference and scoring
helpers from eval_checkpoints_supabase.py via importlib so the prompt
template and scoring pipeline stay identical.

Outputs
-------
reports/incremental/
    forgetting_analysis.txt    — per-metric C1 vs C4 means + delta
    forgetting_analysis_raw.json (companion) — raw C4 outputs for the 100 records

Usage
-----
    python experiment/evaluation/forgetting_analysis.py
"""

import os
import sys
import json
import logging
from pathlib import Path

# Match the Windows / Triton compatibility setup of the parent eval script
os.environ.setdefault("TORCHDYNAMO_DISABLE", "1")
os.environ.setdefault("TORCHINDUCTOR_COMPILE_THREADS", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Path setup — make project root and the evaluation dir importable
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))
eval_dir = str(project_root / "02_distillation" / "evaluation_suite")
if eval_dir not in sys.path:
    sys.path.insert(0, eval_dir)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# Reuse everything from the canonical eval script — same prompt format,
# same scoring, same model-loading, same metric set.
from importlib import import_module                       # noqa: E402
ev = import_module("eval_checkpoints_supabase")

from src.database.supabase_client import get_supabase_client  # noqa: E402

REPORTS_DIR = ev.REPORTS_DIR
RAW_PATH = REPORTS_DIR / "checkpoint_eval_supabase_raw.json"
OUT_TXT = REPORTS_DIR / "forgetting_analysis.txt"
OUT_JSON = REPORTS_DIR / "forgetting_analysis_raw.json"

C4_ADAPTER = ev.CHECKPOINTS["C4"]
METRICS = ev.METRICS
METRIC_LABELS = ev.METRIC_LABELS


def load_c1_records_from_raw() -> list[dict]:
    """
    Pull the 100 records that C1 was evaluated on from the persisted
    raw.json. We filter for ckpt='C1' AND record-checkpoint==1 because
    raw.json contains C1's scores across all four checkpoint slices —
    we only want the slice where source-checkpoint == 1.
    """
    if not RAW_PATH.exists():
        log.error(
            f"Required input not found: {RAW_PATH}. Run "
            f"eval_checkpoints_supabase.py first."
        )
        sys.exit(1)

    with open(RAW_PATH, "r", encoding="utf-8") as f:
        all_raw = json.load(f)

    c1_slice = [
        r for r in all_raw
        if r.get("ckpt") == "C1" and r.get("checkpoint") == 1
    ]
    if len(c1_slice) != 100:
        log.warning(
            f"Expected 100 C1-on-checkpoint=1 records, got {len(c1_slice)}. "
            f"Continuing anyway."
        )
    log.info(f"Loaded {len(c1_slice)} C1 baseline records from raw.json")
    return c1_slice


def hydrate_records(ids: list[int]) -> list[dict]:
    """
    Re-fetch input / context / sevenb / checkpoint columns for the
    given ids. We do this rather than relying on the truncated copies
    in raw.json (input/reference were stored with [:200]/[:400] limits)
    so inference uses the full prompt that C1 saw.
    """
    client = get_supabase_client()
    resp = (
        client.table("modelcomp_50k")
        .select("id, input, context, sevenb, checkpoint")
        .in_("id", ids)
        .execute()
    )
    rows = list(resp.data or [])
    by_id = {r["id"]: r for r in rows}
    # Preserve the original order (matches raw.json ascending-id ordering).
    ordered = [by_id[i] for i in ids if i in by_id]
    log.info(
        f"Hydrated {len(ordered)}/{len(ids)} records from Supabase "
        f"(non-empty sevenb: "
        f"{sum(1 for r in ordered if (r.get('sevenb') or '').strip())})"
    )
    if len(ordered) != len(ids):
        missing = set(ids) - set(by_id.keys())
        log.warning(f"Missing ids after hydrate: {sorted(missing)[:10]}...")
    return ordered


def build_comparison_table(
    c1_records: list[dict], c4_records: list[dict]
) -> tuple[str, dict]:
    """
    Given matched C1 and C4 result lists (same ids in same order),
    compute per-metric means + deltas and format as a fixed-width table.
    Returns (table_string, summary_dict).
    """
    assert len(c1_records) == len(c4_records), (
        f"Record count mismatch: C1={len(c1_records)} C4={len(c4_records)}"
    )
    n = len(c1_records)

    summary: dict[str, dict[str, float]] = {}
    for m in METRICS:
        c1_mean = sum(r["metrics"][m] for r in c1_records) / n
        c4_mean = sum(r["metrics"][m] for r in c4_records) / n
        summary[m] = {
            "c1_mean": round(c1_mean, 4),
            "c4_mean": round(c4_mean, 4),
            "delta":   round(c4_mean - c1_mean, 4),
        }

    label_w = 26
    col_w = 10
    header = (
        f"{'Metric':<{label_w}}  "
        f"{'C1_mean':>{col_w}}  {'C4_mean':>{col_w}}  "
        f"{'Δ (C4-C1)':>{col_w}}  {'Direction':<10}"
    )
    sep = "-" * len(header)

    lines = [
        "FORGETTING ANALYSIS — C4 adapter on C1's evaluation slice",
        "=" * len(header),
        f"Eval set: 100 records from modelcomp_50k checkpoint=1 "
        f"(highest-id slice C1 was originally evaluated on)",
        "",
        sep,
        header,
        sep,
    ]

    # For Hallucination, lower is better — flip the direction interpretation.
    LOWER_IS_BETTER = {"hallucination"}

    for m in METRICS:
        s = summary[m]
        delta = s["delta"]
        if abs(delta) < 0.001:
            direction = "≈ same"
        elif m in LOWER_IS_BETTER:
            direction = "improved" if delta < 0 else "FORGOT"
        else:
            direction = "improved" if delta > 0 else "FORGOT"
        lines.append(
            f"{METRIC_LABELS[m]:<{label_w}}  "
            f"{s['c1_mean']:>{col_w}.4f}  {s['c4_mean']:>{col_w}.4f}  "
            f"{delta:>+{col_w}.4f}  {direction:<10}"
        )

    lines.append(sep)
    lines.append("")
    lines.append(f"  n = {n} records")
    lines.append(
        f"  C1 baseline source: ckpt='C1', checkpoint==1 in "
        f"{RAW_PATH.name}"
    )
    lines.append("  Note: for Hallucination, lower is better — sign convention flipped.")

    return "\n".join(lines), summary


def main() -> None:
    log.info("=" * 60)
    log.info("FORGETTING ANALYSIS — C4 vs C1 on C1's eval slice")
    log.info("=" * 60)

    # ── Load C1 baseline from raw.json ─────────────────────────────────────
    c1_records = load_c1_records_from_raw()
    if not c1_records:
        log.error("No C1 records found in raw.json — aborting.")
        sys.exit(1)

    ids_in_order = [r["id"] for r in c1_records]
    log.info(f"  id range: {min(ids_in_order)}..{max(ids_in_order)}")

    # ── Hydrate full input/context/sevenb from Supabase ────────────────────
    val_data = hydrate_records(ids_in_order)
    if len(val_data) != len(c1_records):
        log.error(
            f"Hydration size mismatch ({len(val_data)} vs "
            f"{len(c1_records)}) — aborting."
        )
        sys.exit(1)

    # Sanity check — same as parent script
    bad = [r for r in val_data if not (r.get("sevenb") or "").strip()]
    if bad:
        log.warning(f"⚠  {len(bad)} records have empty sevenb")

    # ── Run C4 evaluation (this is the only model load) ────────────────────
    if not Path(C4_ADAPTER).exists():
        log.error(f"C4 adapter missing: {C4_ADAPTER} — aborting.")
        sys.exit(1)

    c4_records = ev.evaluate_one_checkpoint("C4", C4_ADAPTER, val_data)
    if len(c4_records) != len(c1_records):
        log.error(
            f"C4 produced {len(c4_records)} results for "
            f"{len(c1_records)} inputs — aborting."
        )
        sys.exit(1)

    # ── Re-align by id (just in case batched inference reordered) ──────────
    c1_by_id = {r["id"]: r for r in c1_records}
    c4_by_id = {r["id"]: r for r in c4_records}
    common_ids = [i for i in ids_in_order if i in c1_by_id and i in c4_by_id]
    c1_aligned = [c1_by_id[i] for i in common_ids]
    c4_aligned = [c4_by_id[i] for i in common_ids]

    # ── Persist raw C4 results for reproducibility ─────────────────────────
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "description": (
                    "C4 adapter run on the 100 records from "
                    "modelcomp_50k checkpoint=1 that C1 was originally "
                    "evaluated on (forgetting check)."
                ),
                "n": len(c4_aligned),
                "records": c4_aligned,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )
    log.info(f"Raw C4 outputs saved: {OUT_JSON}")

    # ── Build + write comparison table ─────────────────────────────────────
    table_str, summary = build_comparison_table(c1_aligned, c4_aligned)
    with open(OUT_TXT, "w", encoding="utf-8") as f:
        f.write(table_str + "\n")

    print()
    print(table_str)
    print()
    log.info(f"Comparison saved: {OUT_TXT}")
    log.info(f"Summary: {json.dumps(summary, indent=2)}")


if __name__ == "__main__":
    main()
