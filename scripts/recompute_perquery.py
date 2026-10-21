#!/usr/bin/env python3
"""
Phase A, step 1: recompute every per-query score from the stored generations,
using the user's own metric code (06_eval_metrics.py, copied as eval_metrics06.py)
and the same ROUGE/BLEU calls as eval_retention_matrix.py.

Inputs : matrix_raw.json, c1..c4_eval_slice.json,
         heldout_slice.json  (all in reports/retention/)
Output : out/perquery_recomputed.json  (one row per adapter x cohort x id, with
         stored and recomputed values, incl. task_success and faithfulness that
         the stored files do not keep)
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import importlib.util
_spec = importlib.util.spec_from_file_location("em", ROOT / "02_distillation" / "evaluation_suite" / "06_eval_metrics.py")
em = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(em)
from rouge_score import rouge_scorer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

U = ROOT / "reports" / "retention"
OUT = U
OUT.mkdir(exist_ok=True)

METRICS9 = ["structured_correctness", "instruction_following", "coverage",
            "hallucination", "context_grounding", "conciseness",
            "rouge1", "rougel", "bleu"]

slice_files = {
    "C1": "c1_eval_slice.json",
    "C2": "c2_eval_slice.json",
    "C3": "c3_eval_slice.json",
    "C4": "c4_eval_slice.json",
    "H": "heldout_slice.json",
}
slices = {k: {r["id"]: r for r in json.load(open(U / f, encoding="utf-8"))}
          for k, f in slice_files.items()}

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
smooth = SmoothingFunction().method1


def score(pred, rec):
    """Mirror of calculate_metrics() in eval_retention_matrix.py (task_label general_qa)."""
    reference = (rec.get("sevenb") or "").strip()
    instruction = (rec.get("input") or "").strip()
    context = (rec.get("context") or "").strip()
    keys = METRICS9 + ["overall", "task_success", "faithfulness"]
    if not pred or not reference:
        return {k: 0.0 for k in keys}
    ev = em.evaluate_single_output(instruction=instruction, student_output=pred,
                                   teacher_output=reference, context=context,
                                   task_label="general_qa")
    rs = scorer.score(reference, pred)
    try:
        bleu = sentence_bleu([reference.split()], pred.split(), smoothing_function=smooth)
    except Exception:
        bleu = 0.0
    return {
        "overall": ev["overall_score"],
        "structured_correctness": ev["structured_correctness"],
        "instruction_following": ev["instruction_following"],
        "coverage": ev["coverage"],
        "hallucination": ev["hallucination"],
        "context_grounding": ev["context_grounding"],
        "conciseness": ev["conciseness"],
        "task_success": ev["task_success"],
        "faithfulness": ev["faithfulness"],
        "rouge1": rs["rouge1"].fmeasure,
        "rougel": rs["rougeL"].fmeasure,
        "bleu": bleu,
    }


rows = []
raw = json.load(open(U / "matrix_raw.json", encoding="utf-8"))
for r in raw:
    rec = slices[r["eval_slice"]][r["id"]]
    rc = score(r["generated"], rec)
    rows.append({"source": "matrix_raw", "adapter": r["adapter"], "cohort": r["eval_slice"],
                 "id": r["id"], "stored": {**r["metrics"], "overall": r["overall"]},
                 "recomputed": rc, "hit_limit": r.get("hit_limit"),
                 "n_new_tokens": r.get("n_new_tokens"), "generated": r["generated"]})

# agreement check: stored values are rounded to 4 dp in matrix_raw
worst = {}
for row in rows:
    for k in METRICS9 + ["overall"]:
        d = abs(row["stored"][k] - row["recomputed"][k])
        key = (row["source"], k)
        worst[key] = max(worst.get(key, 0.0), d)
print("Max |stored - recomputed| per source/metric:")
for (s, k), v in sorted(worst.items()):
    print(f"  {s:16s} {k:24s} {v:.6f}")

json.dump(rows, open(OUT / "perquery_recomputed.json", "w"), indent=1)
print(f"rows: {len(rows)} -> {OUT / 'perquery_recomputed.json'}")
