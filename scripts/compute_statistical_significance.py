#!/usr/bin/env python3
"""
=============================================================================
Statistical Significance & Rigorous Uncertainty Analysis for C3B-Adapter
Venue: ICSE 2027 (NIER Track)
=============================================================================
Computes:
1. Paired t-tests and Wilcoxon signed-rank tests for historical retention degradation:
   - C1 vs C2 on Slice C1 (N=100)
   - C1 vs C3 on Slice C1 (N=100)
   - C1 vs C4 on Slice C1 (N=100)
2. Statistical breakdown across all 9 deterministic metrics.
3. 95% Confidence Intervals (CI) for retention decay and replay mitigation recovery.
4. Generates both JSON audit data and Markdown summary for LaTeX integration.
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MATRIX_RAW_FILE = PROJECT_ROOT / "reports" / "retention" / "matrix_raw.json"
REPLAY_COMP_FILE = PROJECT_ROOT / "reports" / "retention" / "replay_comparison.json"
OUTPUT_JSON_FILE = PROJECT_ROOT / "reports" / "retention" / "statistical_significance_report.json"
OUTPUT_MD_FILE = PROJECT_ROOT / "reports" / "retention" / "statistical_significance_report.md"

METRICS = [
    "overall",
    "instruction_following",
    "structured_correctness",
    "coverage",
    "hallucination",
    "context_grounding",
    "conciseness",
    "rouge1",
    "rougel",
    "bleu",
]

def analyze_paired_slice(records_a, records_b, label_a, label_b, slice_name="C1"):
    common_ids = sorted(list(set(records_a.keys()) & set(records_b.keys())))
    n = len(common_ids)
    results = {
        "comparison": f"{label_a} vs {label_b} on Slice {slice_name}",
        "n_samples": n,
        "metrics": {}
    }

    for m in METRICS:
        va = [records_a[i]["metrics"][m] if m != "overall" else records_a[i]["overall"] for i in common_ids]
        vb = [records_b[i]["metrics"][m] if m != "overall" else records_b[i]["overall"] for i in common_ids]
        
        diff = np.array(vb) - np.array(va)
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if n > 1 else 0.0
        se = float(std_diff / np.sqrt(n)) if n > 0 else 0.0
        ci95 = [float(mean_diff - 1.96 * se), float(mean_diff + 1.96 * se)]

        t_stat, p_val_t = stats.ttest_rel(vb, va)
        try:
            w_stat, p_val_w = stats.wilcoxon(vb, va)
        except Exception:
            w_stat, p_val_w = 0.0, 1.0

        results["metrics"][m] = {
            "mean_a": float(np.mean(va)),
            "mean_b": float(np.mean(vb)),
            "mean_delta": mean_diff,
            "std_delta": std_diff,
            "std_error": se,
            "ci95": ci95,
            "t_statistic": float(t_stat),
            "p_value_t": float(p_val_t),
            "wilcoxon_stat": float(w_stat),
            "p_value_wilcoxon": float(p_val_w),
            "statistically_significant_05": bool(p_val_t < 0.05 or p_val_w < 0.05),
        }

    return results

def main():
    print(f"Loading raw matrix data from: {MATRIX_RAW_FILE}")
    with open(MATRIX_RAW_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Slice C1 records
    c1_records = {d["id"]: d for d in data if d["adapter"] == "C1" and d["eval_slice"] == "C1"}
    c2_records = {d["id"]: d for d in data if d["adapter"] == "C2" and d["eval_slice"] == "C1"}
    c3_records = {d["id"]: d for d in data if d["adapter"] == "C3" and d["eval_slice"] == "C1"}
    c4_records = {d["id"]: d for d in data if d["adapter"] == "C4" and d["eval_slice"] == "C1"}

    analysis_c1_c2 = analyze_paired_slice(c1_records, c2_records, "C1", "C2", "C1")
    analysis_c1_c3 = analyze_paired_slice(c1_records, c3_records, "C1", "C3", "C1")
    analysis_c1_c4 = analyze_paired_slice(c1_records, c4_records, "C1", "C4", "C1")

    # Load replay comparison
    with open(REPLAY_COMP_FILE, "r", encoding="utf-8") as f:
        replay_comp = json.load(f)

    output = {
        "statistical_analysis": {
            "slice_c1_c1_vs_c2": analysis_c1_c2,
            "slice_c1_c1_vs_c3": analysis_c1_c3,
            "slice_c1_c1_vs_c4": analysis_c1_c4,
            "replay_summary": replay_comp.get("comparison_table", []),
        }
    }

    with open(OUTPUT_JSON_FILE, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    # Generate Markdown Table
    md_lines = [
        "# Statistical Significance & Uncertainty Audit Report",
        "",
        "## 1. Longitudinal Retention Decay on Historical Slice C1 (N = 100)",
        "",
        "| Metric | Baseline (C1) | C2-Original | $\\Delta$ (C2 - C1) | 95% Conf. Interval | $p$-value (t-test) | $p$-value (Wilcoxon) | Sig? |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for m in METRICS:
        info = analysis_c1_c2["metrics"][m]
        sig_str = "**YES** (p < 0.05)" if info["statistically_significant_05"] else "No"
        ci_str = f"[{info['ci95'][0]:+.4f}, {info['ci95'][1]:+.4f}]"
        md_lines.append(
            f"| `{m}` | {info['mean_a']:.4f} | {info['mean_b']:.4f} | {info['mean_delta']:+.4f} | {ci_str} | {info['p_value_t']:.4f} | {info['p_value_wilcoxon']:.4f} | {sig_str} |"
        )

    md_lines.extend([
        "",
        "## 2. Long-Horizon Degradation: Baseline C1 vs Checkpoint C4 (N = 100)",
        "",
        "| Metric | Baseline (C1) | Checkpoint C4 | $\\Delta$ (C4 - C1) | 95% Conf. Interval | $p$-value (t-test) | $p$-value (Wilcoxon) | Sig? |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |",
    ])

    for m in METRICS:
        info = analysis_c1_c4["metrics"][m]
        sig_str = "**YES** (p < 0.05)" if info["statistically_significant_05"] else "No"
        ci_str = f"[{info['ci95'][0]:+.4f}, {info['ci95'][1]:+.4f}]"
        md_lines.append(
            f"| `{m}` | {info['mean_a']:.4f} | {info['mean_b']:.4f} | {info['mean_delta']:+.4f} | {ci_str} | {info['p_value_t']:.4f} | {info['p_value_wilcoxon']:.4f} | {sig_str} |"
        )

    with open(OUTPUT_MD_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines) + "\n")

    print(f"Statistical audit successfully written to:")
    print(f"  - JSON: {OUTPUT_JSON_FILE}")
    print(f"  - MD:   {OUTPUT_MD_FILE}")

if __name__ == "__main__":
    main()

