#!/usr/bin/env python3
"""
No-GPU gate check on saved per-query outputs (canary = cohort C1, N = 100).

Rules (unchanged from the paper's code thresholds):
  G1  mean instruction_following >= 0.95 and mean structured_correctness >= 0.50
  G2  R = S(cand)/S(inc) >= 0.98   and   dHallu = H(cand) - H(inc) <= +0.015,
      with H(inc) the MEASURED incumbent mean (not a default constant)
  G3  mean per-query latency <= 3.0 s and peak VRAM <= 4.0 GB, read from
      phaseB_log.jsonl if measured; otherwise reported as "not measured"
Uncertainty: bootstrap of R, 10,000 paired query resamples, seed 42, percentile 95% CI,
and P(R* < 0.98).
Three-state decision:
  BLOCK         G1 fails, or dHallu fails, or the whole R CI is below 0.98
  PASS          G1 and dHallu pass and the whole R CI is >= 0.98
  INCONCLUSIVE  the R CI straddles 0.98
  The paper's original point rule (R >= 0.98) is reported next to it.
  Flag 'ci_includes_1' marks cases with no evidence of any change.
Gate-logic time: each decision is timed over 30 repeats (time.perf_counter).

Usage (PowerShell, from the project root):
  python scripts\\gate_check.py --matrix reports\\retention\\matrix_raw.json `
         --out reports\\retention\\gate_decisions.json
"""
import json, time, argparse, statistics
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--matrix", required=True, help="matrix_raw.json")
ap.add_argument("--phaseb", default=None, help="folder with perquery_*.json and phaseB_log.jsonl")
ap.add_argument("--replay-unsloth", default=None, help="optional replay_perquery_canary.json (batch-32 Unsloth run; flagged MIXED)")
ap.add_argument("--out", required=True)
ap.add_argument("--B", type=int, default=10000)
args = ap.parse_args()

R_MIN, HALLU_MAX, IF_MIN, SC_MIN, LAT_MAX, VRAM_MAX = 0.98, 0.015, 0.95, 0.50, 3.0, 4.0
data, setup = {}, {}

def put(label, rid, overall, m, src):
    data.setdefault(label, {})[rid] = dict(overall=overall, hallucination=m["hallucination"],
                                           instruction_following=m["instruction_following"],
                                           structured_correctness=m["structured_correctness"])
    setup[label] = src

for r in json.load(open(args.matrix, encoding="utf-8")):
    if r["eval_slice"] == "C1":
        put(r["adapter"], r["id"], r["overall"], r["metrics"], "matrix (HF+PEFT, batch 4)")
ops = {}
if args.phaseb:
    pb = Path(args.phaseb)
    for f in sorted(pb.glob("perquery_*.json")):
        for r in json.load(open(f, encoding="utf-8")):
            put(r["adapter"], r["id"], r["overall"], r["metrics"], "phaseB (HF+PEFT, batch 4)")
    lg = pb / "phaseB_log.jsonl"
    if lg.exists():
        for line in open(lg, encoding="utf-8"):
            e = json.loads(line)
            if e.get("event") == "eval_done":
                ops[e["label"]] = dict(latency=e.get("per_query_seconds_mean"), vram=e.get("peak_vram_gb"))
if args.replay_unsloth:
    for r in json.load(open(args.replay_unsloth, encoding="utf-8"))["per_query_evaluations"]:
        put("C2R_UNSLOTH", r["id"], r["overall"], r["metrics"], "Unsloth batch 32 (MIXED vs matrix)")


def decide(cand, inc, B=args.B, seed=42):
    ids = sorted(set(data[cand]) & set(data[inc]))
    x = np.array([data[cand][i]["overall"] for i in ids]); y = np.array([data[inc][i]["overall"] for i in ids])
    mean = lambda lab, k: float(np.mean([data[lab][i][k] for i in ids]))
    R = float(x.mean() / y.mean())
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(B, len(ids)))
    Rb = x[idx].mean(1) / y[idx].mean(1)
    lo, hi = float(np.percentile(Rb, 2.5)), float(np.percentile(Rb, 97.5))
    d_h = mean(cand, "hallucination") - mean(inc, "hallucination")
    g1 = mean(cand, "instruction_following") >= IF_MIN and mean(cand, "structured_correctness") >= SC_MIN
    g2_h = d_h <= HALLU_MAX
    op = ops.get(cand)
    g3 = None if not op or op["latency"] is None or op["vram"] is None else (op["latency"] <= LAT_MAX and op["vram"] <= VRAM_MAX)
    if not g1 or not g2_h or hi < R_MIN or g3 is False:
        dec = "BLOCK"
    elif lo >= R_MIN:
        dec = "PASS"
    else:
        dec = "INCONCLUSIVE"
    return dict(candidate=cand, incumbent=inc, n=len(ids), S_cand=float(x.mean()), S_inc=float(y.mean()),
                R=R, R_ci95=[lo, hi], P_Rstar_below_098=float((Rb < R_MIN).mean()),
                ci_includes_1=bool(lo <= 1.0 <= hi), H_cand=mean(cand, "hallucination"),
                H_inc=mean(inc, "hallucination"), dHallu=d_h, dHallu_pass=g2_h,
                IF_cand=mean(cand, "instruction_following"), SC_cand=mean(cand, "structured_correctness"),
                G1_pass=g1, G3=("not measured" if g3 is None else g3), G3_inputs=op,
                point_rule=("PASS" if (g1 and g2_h and R >= R_MIN) else "BLOCK"),
                decision=dec, setup_cand=setup[cand], setup_inc=setup[inc])


pairs = [("C2", "C1"), ("C3", "C2"), ("C4", "C3"),          # vs incumbent
         ("C3", "C1"), ("C4", "C1")]                         # vs C1 (drift)
for extra in ["C2R", "C1_RERUN", "C1_X05", "C1_NOISE025", "C1_NOISE100", "C2R_UNSLOTH"]:
    if extra in data:
        pairs.append((extra, "C1"))

results = []
for cand, inc in pairs:
    t = []
    for _ in range(30):
        t0 = time.perf_counter(); res = decide(cand, inc); t.append((time.perf_counter() - t0) * 1000)
    res["gate_logic_ms"] = dict(mean=statistics.mean(t), p50=statistics.median(t),
                                p95=sorted(t)[int(0.95 * 29)], repeats=30, bootstrap_B=args.B)
    results.append(res)

json.dump(results, open(args.out, "w"), indent=2)
print(f"{'cand':12s}{'inc':5s}{'R':>9s}{'95% CI':>20s}{'P(R*<.98)':>11s}{'dHallu':>9s}{'G1':>4s}{'point':>7s}{'decision':>14s}{'ms':>8s}")
for r in results:
    print(f"{r['candidate']:12s}{r['incumbent']:5s}{r['R']:9.4f}  [{r['R_ci95'][0]:.4f}, {r['R_ci95'][1]:.4f}]"
          f"{r['P_Rstar_below_098']:11.3f}{r['dHallu']:+9.4f}{'Y' if r['G1_pass'] else 'N':>4s}{r['point_rule']:>7s}"
          f"{r['decision']:>14s}{r['gate_logic_ms']['mean']:8.1f}"
          + ("   <- MIXED setup" if "MIXED" in r["setup_cand"] else ""))
print("saved", args.out)
