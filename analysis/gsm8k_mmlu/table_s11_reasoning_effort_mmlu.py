"""Supplementary Table 11, MMLU-Pro rows: saving of ReD for gpt-oss-20b at low and medium reasoning effort.

Saving = (budget that solve-to-completion (STC) needs to reach a target coverage) / (budget of ReD),
for targets of 25%, 50% and 75% of N = 500 (the first 500 MMLU-Pro problems), with the budget counted
in attempts or in output tokens. k = 50 attempts per problem. The budget to reach a target is averaged
over 200 realizations (random row and column shuffles of the N x k matrices), seed 0.
STC and ReD (tau = 1) are simulated on the same shuffle in each realization (coverage_simulation.py).

Adapted from make_report.py of the ReD4LLMs repository (the code that produced the paper's numbers),
keeping only the table that is in the paper.

Run (from anywhere):
    python analysis/gsm8k_mmlu/table_s11_reasoning_effort_mmlu.py
Inputs: data/mmlu-pro-500/openai_gpt-oss-20b_{low,medium}_matrix_matrices.npz
        (dense matrices built from the *_matrix.jsonl files by inference/gsm8k_mmlu/build_matrices.py)
Output: outputs/tables/table_s11_reasoning_effort_mmlu.md
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from coverage_simulation import run_monte_carlo  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data", "mmlu-pro-500")
TARGETS_FRAC = (0.25, 0.5, 0.75)


def load_effort(npz_path):
    d = np.load(npz_path)
    return {
        "correct": d["correct"],
        "out_tokens": d["out_tokens"].astype(np.float64),
        "seen": d["seen"],
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--R", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "tables", "table_s11_reasoning_effort_mmlu.md"))
    a = ap.parse_args()

    lines = ["**Supplementary Table 11, MMLU-Pro rows.** Saving of ReD (STC budget / ReD budget) "
             f"for gpt-oss-20b, N=500, k=50, {a.R} realizations.", "",
             "| Effort | Attempts 25% | Attempts 50% | Attempts 75% | Output tokens 25% | Output tokens 50% | Output tokens 75% |",
             "|---|---|---|---|---|---|---|"]
    for effort in ["low", "medium"]:
        mats = load_effort(os.path.join(DATA, f"openai_gpt-oss-20b_{effort}_matrix_matrices.npz"))
        n_missing = int((~mats["seen"]).sum())
        print(f"{effort}: matrix {mats['correct'].shape}, {n_missing} missing cells (counted as failures)")
        sim = run_monte_carlo(mats["correct"], mats["out_tokens"], R=a.R, seed=a.seed,
                              coverage_targets_frac=TARGETS_FRAC)
        att = [s / r for s, r in zip(sim["stc"]["cross_attempts"], sim["red"]["cross_attempts"])]
        tok = [s / r for s, r in zip(sim["stc"]["cross_tokens"], sim["red"]["cross_tokens"])]
        lines.append(f"| {effort} | " + " | ".join(f"{v:.1f}" for v in att + tok) + " |"
                     + "  <!-- unrounded: " + ", ".join(f"{v:.2f}" for v in att + tok) + " -->")
    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(md)
    print(md)
    print(f"[wrote {a.out}]")


if __name__ == "__main__":
    main()
