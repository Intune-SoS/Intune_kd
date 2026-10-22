#!/usr/bin/env python3
"""
Final numbers for main_final.tex. Every number in the paper comes from this script's
output (final_numbers.json / f1_stdout.txt).

Sources (reports/retention/):
  reports_retention/matrix_raw.json      matrix run, HF+PEFT, batch 4 (Base, C1-C4; cohorts C1-C4, H)
  reports_retention/replay_comparison.json  summary only (Unsloth batch 32): C2 0.5554, C2-Replay 0.5869
  reports_retention/replay_perquery_canary.json  C2-Replay per-query (Unsloth batch 32)
  perquery_recomputed.json (from recompute_perquery.py)        task_success / faithfulness rescored from the matrix
                                          generations with 06_eval_metrics.py (script a1)
Rules: Wilcoxon signed-rank (scipy, zero_method='wilcox', two-sided) primary, paired t
secondary; BH over the 9 stored metric rows of one comparison; bootstrap of R:
10,000 paired resamples, seed 42, percentile 95% CI. Decision: BLOCK if G1 fails, dHallu
> +0.015, or CI upper < 0.98; PASS if CI lower >= 0.98; else INCONCLUSIVE.
"""
import json
from pathlib import Path
import numpy as np
from scipy import stats

B = Path(__file__).resolve().parents[1] / "reports" / "retention"
OUT = B
raw = json.load(open(B / "matrix_raw.json", encoding="utf-8"))
rc = json.load(open(B / "replay_comparison.json"))
recomp = json.load(open(B / "perquery_recomputed.json"))

M9 = ["instruction_following", "structured_correctness", "coverage", "hallucination",
      "context_grounding", "conciseness", "rouge1", "rougel", "bleu"]
D = {}
for r in raw:
    D.setdefault((r["adapter"], r["eval_slice"]), {})[r["id"]] = {**r["metrics"], "overall": r["overall"],
                                                                   "hit_limit": r["hit_limit"]}
for r in recomp:   # add task_success, faithfulness (matrix rows only)
    if r["source"] == "matrix_raw":
        D[(r["adapter"], r["cohort"])][r["id"]]["task_success"] = r["recomputed"]["task_success"]
        D[(r["adapter"], r["cohort"])][r["id"]]["faithfulness"] = r["recomputed"]["faithfulness"]

N = {}
def v(a, c, m, ids=None):
    ids = ids or sorted(D[(a, c)]); return np.array([D[(a, c)][i][m] for i in ids], float)
def S(a, c, ids=None): return float(v(a, c, "overall", ids).mean())
def bh(p):
    p = np.asarray(p); n = len(p); o = np.argsort(p); q = np.empty(n); prev = 1.0
    for k in range(n, 0, -1):
        i = o[k - 1]; prev = min(prev, p[i] * n / k); q[i] = prev
    return q
def paired(a, b, c, m, ids=None):
    ids = ids or sorted(set(D[(a, c)]) & set(D[(b, c)]))
    x, y = v(a, c, m, ids), v(b, c, m, ids); d = y - x
    pw = float(stats.wilcoxon(y, x).pvalue) if (d != 0).any() else 1.0
    pt = float(stats.ttest_rel(y, x).pvalue) if (d != 0).any() else 1.0
    return dict(delta=float(d.mean()), p_w=pw, p_t=pt, n_dec=int((d < 0).sum()), n_inc=int((d > 0).sum()),
                se=float(d.std(ddof=1) / np.sqrt(len(d))))

A = ["BASE", "C1", "C2", "C3", "C4"]; COH = ["C1", "C2", "C3", "C4", "H"]
N["table1"] = {a: {c: S(a, c) for c in COH} for a in A}
N["n_per_cohort"] = {c: len(D[("C1", c)]) for c in COH}

# Table II: canary per-metric means + C2 vs C1 tests, BH over 9 rows
EXTRA = ["task_success", "faithfulness"]
N["table2_means"] = {a: {m: float(v(a, "C1", m).mean()) for m in M9 + EXTRA + ["overall"]} for a in A}
t2 = {m: paired("C1", "C2", "C1", m) for m in M9 + EXTRA + ["overall"]}
qw = bh([t2[m]["p_w"] for m in M9]); qt = bh([t2[m]["p_t"] for m in M9])
for m, a1, a2 in zip(M9, qw, qt): t2[m]["q_w"] = float(a1); t2[m]["q_t"] = float(a2)
N["table2_C2_vs_C1"] = t2

# Gate decisions (Table III)
def decide(cand, inc, Bn=10000, seed=42):
    ids = sorted(set(D[(cand, "C1")]) & set(D[(inc, "C1")]))
    x, y = v(cand, "C1", "overall", ids), v(inc, "C1", "overall", ids)
    idx = np.random.default_rng(seed).integers(0, len(ids), size=(Bn, len(ids)))
    Rb = x[idx].mean(1) / y[idx].mean(1)
    lo, hi = float(np.percentile(Rb, 2.5)), float(np.percentile(Rb, 97.5))
    R = float(x.mean() / y.mean())
    dh = float(v(cand, "C1", "hallucination", ids).mean() - v(inc, "C1", "hallucination", ids).mean())
    g1 = v(cand, "C1", "instruction_following").mean() >= 0.95 and v(cand, "C1", "structured_correctness").mean() >= 0.50
    dec = "BLOCK" if (not g1 or dh > 0.015 or hi < 0.98) else ("PASS" if lo >= 0.98 else "INCONCLUSIVE")
    point = "PASS" if (g1 and dh <= 0.015 and R >= 0.98) else "BLOCK"
    pr = paired(inc, cand, "C1", "overall", ids)
    d = x - y
    rho = {e: float((d < -e).mean()) for e in (0.0, 0.01, 0.05)}
    return dict(R=R, ci=[lo, hi], P_below=float((Rb < 0.98).mean()), dHallu=dh, G1=bool(g1),
                point_rule=point, decision=dec, delta=pr["delta"], p_w=pr["p_w"], p_t=pr["p_t"],
                n_dec=pr["n_dec"], n_inc=pr["n_inc"], se=pr["se"], rho=rho)
pairs = [("C2", "C1"), ("C3", "C2"), ("C4", "C3"), ("C3", "C1"), ("C4", "C1")]
N["gate"] = {f"{a}_vs_{b}": decide(a, b) for a, b in pairs}
# aggregate-only baseline: candidate vs incumbent on the incoming cohort
N["aggregate"] = {f"{a}_vs_{b}": dict(cohort=a, delta=S(a, a) - S(b, a), cand=S(a, a), inc=S(b, a))
                  for a, b in [("C2", "C1"), ("C3", "C2"), ("C4", "C3")]}
# noise
N["allowed_drop"] = 0.02 * S("C1", "C1")
# held-out H (all 300), paired vs C1
N["H"] = {b: paired("C1", b, "H", "overall") for b in ["BASE", "C2", "C3", "C4"]}
# setup sensitivity
c2_b32 = rc["c2_original_mean_scores"]["overall"]
N["setup"] = dict(C2_matrix=S("C2", "C1"), C2_unsloth_b32_summary=c2_b32,
                  diff=c2_b32 - S("C2", "C1"), diff_share_of_allowed=(S("C2", "C1") - c2_b32) / N["allowed_drop"],
                  )
# truncation and SC format proxy
N["hit_limit"] = {a: float(v(a, "C1", "hit_limit").mean()) for a in A}
json.dump(N, open(OUT / "final_numbers.json", "w"), indent=1)

f4 = lambda x: f"{x:.4f}"
print("N per cohort:", N["n_per_cohort"])
print("TABLE I (overall, matrix run)")
for a in A: print(" ", a, " ".join(f4(N["table1"][a][c]) for c in COH))
print("TABLE II canary means (C1 cohort)")
for m in M9 + EXTRA + ["overall"]:
    t = t2[m]
    print(f"  {m:24s}", " ".join(f4(N["table2_means"][a][m]) for a in A),
          f"| C2-C1 {t['delta']:+.4f} p_w {t['p_w']:.4f} p_t {t['p_t']:.4f}"
          + (f" q_w {t['q_w']:.4f} q_t {t['q_t']:.4f}" if "q_w" in t else ""))
print("GATE (canary)")
for k, g in N["gate"].items():
    print(f"  {k}: R {f4(g['R'])} CI [{f4(g['ci'][0])}, {f4(g['ci'][1])}] P {g['P_below']:.3f} dH {g['dHallu']:+.4f} "
          f"G1 {g['G1']} point {g['point_rule']} dec {g['decision']} | delta {g['delta']:+.4f} p_w {g['p_w']:.4f} "
          f"p_t {g['p_t']:.4f} {g['n_dec']}/{g['n_inc']} se {g['se']:.4f} | rho {g['rho'][0.0]:.2f}/{g['rho'][0.01]:.2f}/{g['rho'][0.05]:.2f}")
print("AGGREGATE-ONLY", {k: (f"{x['cand']:.4f}-{x['inc']:.4f}", f"{x['delta']:+.4f}") for k, x in N["aggregate"].items()})
print("allowed drop", f4(N["allowed_drop"]))
print("H (300) vs C1:", {k: (f"{x['delta']:+.4f}", f"p_w {x['p_w']:.4f}", f"p_t {x['p_t']:.4f}") for k, x in N["H"].items()})
s = N["setup"]
print(f"SETUP C2 matrix {s['C2_matrix']:.6f} vs second setup {s['C2_unsloth_b32_summary']} diff {s['diff']:+.4f} share {s['diff_share_of_allowed']:.2f}")
print("hit_limit", {k: round(x, 2) for k, x in N["hit_limit"].items()})
N["R_base_vs_C1"] = S("BASE", "C1") / S("C1", "C1")
json.dump(N, open(OUT / "final_numbers.json", "w"), indent=1)
print("R Base vs C1", f4(N["R_base_vs_C1"]))
# worked example for the paper: query 119413 (largest C2-C1 drop on the canary)
ex = {a: next(r for r in raw if r["adapter"] == a and r["eval_slice"] == "C1" and r["id"] == 119413) for a in ["C1", "C2"]}
N["example_119413"] = dict(C1=ex["C1"]["overall"], C2=ex["C2"]["overall"], delta=ex["C2"]["overall"] - ex["C1"]["overall"],
                           C1_text=ex["C1"]["generated"], C2_text=ex["C2"]["generated"], reference=ex["C1"]["reference"])
json.dump(N, open(OUT / "final_numbers.json", "w"), indent=1)
print("EXAMPLE 119413", N["example_119413"])
