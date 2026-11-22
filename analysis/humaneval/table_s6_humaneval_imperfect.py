"""Supplementary Table 6 (HumanEval): coverage of STC and ReD with imperfect verifiers, llama-3.1-8b.

Coverage of STC and ReD at 1, 3, 5 x N attempts with a perfect verifier and with (FPR, FNR) = (2%, 2%), (2%, 10%),
(5%, 5%), (8%, 8%), (8%, 15%); true coverage (problems whose accepted answer is correct) / N. A correct answer is
rejected with probability FNR and a wrong one accepted with probability FPR; an accepted problem is not attempted again.

Data: data/humaneval/{model}_correctness.csv (164 problems x 100 attempts). 200 realizations; in each, the order of the
problems and of the attempts of each problem is shuffled. Coverage at a budget B is the coverage reached once B
attempts are spent (the last point of each realization's curve with cost <= B), averaged over realizations.

STC, ReD (tau = 1) and the imperfect verifier use simulate_standard / simulate_red of
analysis/gsm8k_mmlu/simulate_strategies.py, the simulator behind the GSM8K and MMLU-Pro tables. The script that
produced the paper's Table 6 was not kept; this script recomputes it from the same data. Because the realizations are
random, the last digit can differ from the paper by one or two standard errors (STC varies much more than ReD).

Run (from anywhere; ~10 s):
    python analysis/humaneval/table_s6_humaneval_imperfect.py
Output: outputs/tables/table_s6_humaneval_imperfect.md
"""
import argparse
import bisect
import contextlib
import csv
import os
import random
import sys

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(REPO, "analysis", "gsm8k_mmlu"))
import simulate_strategies as ss  # noqa: E402

MODELS = {"8b": "llama-3.1-8b-instant"}
NOISE = [(0.0, 0.0), (0.02, 0.02), (0.02, 0.10), (0.05, 0.05), (0.08, 0.08), (0.08, 0.15)]
MULT6 = [1, 3, 5]
PAPER6 = {
    (0.0, 0.0): ([9.3, 24.5, 39.8], [63.7, 84.7, 88.4]), (0.02, 0.02): ([15.5, 42.2, 67.7], [62.3, 83.5, 86.8]),
    (0.02, 0.10): ([14.8, 40.7, 66.1], [57.3, 83.1, 86.5]), (0.05, 0.05): ([21.9, 62.7, 83.9], [60.6, 82.4, 83.8]),
    (0.08, 0.08): ([25.5, 75.1, 81.3], [58.9, 81.1, 81.5]), (0.08, 0.15): ([24.9, 72.4, 81.1], [54.3, 80.5, 81.0]),
}


def load_humaneval(model):
    """{problem: [{"passed": bool}, ...]} from data/humaneval/{model}_correctness.csv."""
    tasks = {}
    with open(os.path.join(REPO, "data", "humaneval", f"{model}_correctness.csv")) as f:
        for row in csv.DictReader(f):
            tid = row["task_id"]
            tasks[tid] = [{"passed": row[c].strip() == "1"} for c in row if c != "task_id" and row[c].strip() != ""]
    return tasks


def cov(curves, n, mult):
    """Coverage once m x n attempts are spent: last point of each curve with cost <= m x n."""
    vals = [[c["y"][bisect.bisect_right(c["x"], m * n) - 1] for c in curves] for m in mult]
    return [(100 * np.mean(v), 100 * np.std(v, ddof=1) / np.sqrt(len(v))) for v in vals]


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--realizations", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "tables", "table_s6_humaneval_imperfect.md"))
    a = ap.parse_args()
    fmt = lambda cells: " | ".join(f"{m:.1f} ± {s:.1f}" for m, s in cells)  # noqa: E731
    diff = lambda cells, p: [round(c[0], 1) - x for c, x in zip(cells, p)]  # noqa: E731
    sgn = lambda d: ", ".join(f"{x:+.1f}".replace("-0.0", "+0.0") for x in d)  # noqa: E731

    # Table 6
    tasks = load_humaneval(MODELS["8b"])
    n = len(tasks)
    L = [f"**Supplementary Table 6. HumanEval, llama-3.1-8b, imperfect verifier** (N={n}, {a.realizations} "
         "realizations; true coverage, mean ± s.e., %)", "", "| FPR | FNR | Method | 1 x N | 3 x N | 5 x N |",
         "|---|---|---|---|---|---|"]
    chk = []
    worst = 0.0
    for i, (fpr, fnr) in enumerate(NOISE):
        random.seed(a.seed + i)
        with contextlib.redirect_stderr(open(os.devnull, "w")):
            stc, _ = ss.simulate_standard(tasks, a.realizations, metric="attempts", fpr=fpr, fnr=fnr)
            red, _ = ss.simulate_red(tasks, a.realizations, metric="attempts", fpr=fpr, fnr=fnr)
        cs, cr = cov(stc, n, MULT6), cov(red, n, MULT6)
        L += [f"| {fpr:.0%} | {fnr:.0%} | STC | {fmt(cs)} |", f"| | | ReD | {fmt(cr)} |"]
        ds, dr = diff(cs, PAPER6[(fpr, fnr)][0]), diff(cr, PAPER6[(fpr, fnr)][1])
        worst = max(worst, max(abs(x) for x in ds + dr))
        chk += [f"| FPR={fpr:.0%} FNR={fnr:.0%}, STC | {sgn(ds)} |",
                f"| FPR={fpr:.0%} FNR={fnr:.0%}, ReD | {sgn(dr)} |"]

    L += ["", "**Check against the paper** (this run minus paper, percentage points)", "", "| Case | differences |",
          "|---|---|"] + chk + ["", f"Largest difference from the paper: {worst:.1f} percentage points."]
    md = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(md)
    print(md)
    print(f"[wrote {a.out}]")


if __name__ == "__main__":
    main()
