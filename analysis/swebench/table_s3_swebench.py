"""Supplementary Table 3: budget that solve-to-completion (STC) and ReD need to reach a target coverage on SWE-bench Lite.

Data: data/swebench/swebench_lite_deepseek_samples.csv, built by inference/swebench/extract_swebench_lite.py from the
samples that Brown et al. released (DeepSeek-Coder-V2-Instruct inside Moatless Tools, up to 250 independent runs per
issue, temperature 1.6). We use the 266 issues without flaky unit tests. One attempt is a full run of the agent.
The few runs that crashed (15 issues have 248 or 249 runs) are counted as failed attempts with zero cost.

For each of 1,000 realizations the order of the issues and of the runs is shuffled (the same shuffle for STC and ReD),
and the budget at which each policy first reaches 5%, 10%, ..., 25% of the 266 issues is recorded; the table gives
its mean over realizations. Saving = STC budget / ReD budget, with the budget counted in attempts, output tokens and
USD (the cost the agent recorded, at $3 / $15 per million input / output tokens). The simulator is
analysis/gsm8k_mmlu/coverage_simulation.py (ReD with tau = 1), run once with the output-token matrix and once with the
USD matrix on the same random shuffles.

The script also prints pass@1 (14.4% in the paper), the mean number of agent steps per attempt (9.9) and the
power-law exponent alpha of 1 - pass@k fitted for k >= 10 (0.12 in the paper).

Run (from anywhere; ~1 min):
    python analysis/swebench/table_s3_swebench.py
Output: outputs/tables/table_s3_swebench.md
"""
import argparse
import csv
import os
import sys
from collections import defaultdict
from math import comb

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "analysis", "gsm8k_mmlu"))
from coverage_simulation import run_monte_carlo  # noqa: E402

K = 250
TARGETS = (0.05, 0.10, 0.15, 0.20, 0.25)
PAPER = {  # target: (STC attempts, ReD attempts, saving attempts, saving output tokens, saving USD)
    0.05: (3340, 95, 35.2, 36.9, 37.8),
    0.10: (6448, 182, 35.5, 37.3, 38.0),
    0.15: (9576, 293, 32.7, 34.2, 34.7),
    0.20: (13044, 536, 24.3, 25.2, 25.5),
    0.25: (16165, 922, 17.5, 18.0, 18.1),
}


def load(path):
    rows = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            rows[r["issue"]].append(r)
    issues = sorted(rows)
    n = len(issues)
    C = np.zeros((n, K), dtype=bool)
    O = np.zeros((n, K), dtype=np.float64)
    U = np.zeros((n, K), dtype=np.float64)
    steps = []
    n_runs = []
    for i, issue in enumerate(issues):
        runs = rows[issue]
        n_runs.append(len(runs))
        for j, r in enumerate(runs):          # crashed runs are missing: their cells stay failed, cost 0
            C[i, j] = r["resolved"] == "1"
            O[i, j] = float(r["output_tokens"])
            U[i, j] = float(r["usd"])
            steps.append(int(r["n_steps"]))
    return C, O, U, np.array(n_runs), float(np.mean(steps))


def unbiased_pass_at_k(n, c, k):
    """Chen et al. (2021): 1 - C(n-c, k) / C(n, k)."""
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data", default=os.path.join(REPO, "data", "swebench", "swebench_lite_deepseek_samples.csv"))
    ap.add_argument("--R", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "tables", "table_s3_swebench.md"))
    a = ap.parse_args()

    C, O, U, n_runs, mean_steps = load(a.data)
    N = C.shape[0]
    pass1 = float(np.mean(C.sum(axis=1) / n_runs))
    ks = np.arange(1, 249)
    pk = np.array([np.mean([unbiased_pass_at_k(int(n), int(c), int(k)) for n, c in zip(n_runs, C.sum(axis=1))])
                   for k in ks])
    fit = ks >= 10
    slope = np.polyfit(np.log(ks[fit]), np.log(1 - pk[fit]), 1)[0]

    sim_tok = run_monte_carlo(C, O, R=a.R, seed=a.seed, coverage_targets_frac=TARGETS)
    sim_usd = run_monte_carlo(C, U, R=a.R, seed=a.seed, coverage_targets_frac=TARGETS)
    assert sim_tok["stc"]["cross_attempts"] == sim_usd["stc"]["cross_attempts"]  # same shuffles

    L = [f"**Supplementary Table 3. SWE-bench Lite** ({N} issues, {a.R} realizations). Budget to reach a target "
         "coverage; saving = STC budget / ReD budget.", "",
         "| Coverage/N | STC attempts | ReD attempts | Saving: attempts | Saving: output tokens | Saving: USD |",
         "|---|---|---|---|---|---|"]
    chk = ["", "**Check against the paper** (this run / paper)", "",
           "| Coverage/N | STC attempts | ReD attempts | attempts | output tokens | USD |", "|---|---|---|---|---|---|"]
    for ti, t in enumerate(TARGETS):
        sa, ra = sim_tok["stc"]["cross_attempts"][ti], sim_tok["red"]["cross_attempts"][ti]
        st, rt = sim_tok["stc"]["cross_tokens"][ti], sim_tok["red"]["cross_tokens"][ti]
        su, ru = sim_usd["stc"]["cross_tokens"][ti], sim_usd["red"]["cross_tokens"][ti]
        vals = (sa, ra, sa / ra, st / rt, su / ru)
        L.append(f"| {t:.0%} | {sa:,.0f} | {ra:,.0f} | {sa / ra:.1f} | {st / rt:.1f} | {su / ru:.1f} |")
        p = PAPER[t]
        chk.append(f"| {t:.0%} | {sa:,.0f} / {p[0]:,} | {ra:,.0f} / {p[1]:,} | {vals[2]:.1f} / {p[2]} | "
                   f"{vals[3]:.1f} / {p[3]} | {vals[4]:.1f} / {p[4]} |")
    L += chk
    L += ["", f"pass@1 = {100 * pass1:.1f}% (paper: 14.4%); mean agent steps per attempt = {mean_steps:.1f} "
              f"(paper: 9.9); alpha from the fit of 1 - pass@k for k >= 10 = {-slope:.2f} (paper: 0.12)."]
    md = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(md)
    print(md)
    print(f"[wrote {a.out}]")


if __name__ == "__main__":
    main()
