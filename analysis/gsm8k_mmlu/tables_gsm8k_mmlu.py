"""Supplementary Tables 1, 2, 4 and 5: coverage of solve-to-completion (STC) and ReD on GSM8K and MMLU-Pro.

  Table 1  GSM8K (N = 1,319), llama-3.1-8b and llama-3.3-70b, perfect verifier, budgets 1, 2, 3, 5 x N attempts
  Table 2  MMLU-Pro (first N = 500), same models, perfect verifier, budgets 1, 3, 5 x N attempts
  Table 4  GSM8K, llama-3.1-8b, perfect and imperfect verifiers, budgets 1, 2, 3 x N attempts
  Table 5  MMLU-Pro, llama-3.1-8b, perfect and imperfect verifiers, budgets 1, 3, 5 x N attempts
  Imperfect verifiers: (FPR, FNR) = (0.02, 0.10), (0.08, 0.15), (0.25, 0.05).

Each model answered every problem k = 100 times (data/gsm8k, data/mmlu-pro-500). STC and ReD are
simulated by simulate_strategies.py (unchanged from the ReD4LLMs repository): in each of 1,000
realizations the order of the problems and of the attempts of each problem is shuffled; STC attempts
each problem until the verifier accepts an answer (or its k attempts run out) before moving on; ReD
(tau = 1) gives every unsolved problem one attempt per round. With an imperfect verifier, a correct
answer is rejected with probability FNR and a wrong answer is accepted with probability FPR; an accepted
problem is not attempted again. The tables report the true coverage (problems whose accepted answer is
correct) / N at a budget of m x N attempts (interpolated linearly between the points of each realization's
curve), averaged over realizations, with its standard error.

The original runs were not seeded, so the last digit can differ from the paper by about one standard
error (~0.2-0.5 percentage points for STC, which depends strongly on where the hard problems fall in
the order; ReD is much less variable). This script is seeded (--seed) and therefore reproducible.

Run (from anywhere; ~10 min on 2 cores):
    python analysis/gsm8k_mmlu/tables_gsm8k_mmlu.py
Output: outputs/tables/tables_s1_s2_s4_s5_gsm8k_mmlu.md
"""
import argparse
import multiprocessing as mp
import os
import random
import sys
import contextlib

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate_strategies as ss  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = {"8b": "llama-3.1-8b-instant", "70b": "llama-3.3-70b-versatile"}
DATASETS = {"gsm8k": "GSM8K", "mmlu-pro-500": "MMLU-Pro"}
BUDGETS = {"gsm8k": [1, 2, 3, 5], "mmlu-pro-500": [1, 3, 5]}
NOISE = [(0.02, 0.1), (0.08, 0.15), (0.25, 0.05)]

# paper values (percent), for the side-by-side check printed at the end
PAPER = {
    ("gsm8k", "8b", 0.0, 0.0): {"STC": [33.6, 66.7, 98.3, 99.0], "ReD": [81.3, 98.1, 99.0, 99.0]},
    ("gsm8k", "70b", 0.0, 0.0): {"STC": [25.1, 49.6, 74.4, 97.5], "ReD": [93.2, 97.0, 97.3, 97.5]},
    ("mmlu-pro-500", "8b", 0.0, 0.0): {"STC": [10.4, 30.3, 50.3], "ReD": [46.5, 76.1, 87.0]},
    ("mmlu-pro-500", "70b", 0.0, 0.0): {"STC": [9.7, 28.2, 46.5], "ReD": [71.8, 87.1, 90.3]},
    ("gsm8k", "8b", 0.02, 0.1): {"STC": [41.5, 83.1, 97.3], "ReD": [73.2, 96.9, 97.3]},
    ("gsm8k", "8b", 0.08, 0.15): {"STC": [52.1, 94.4, 94.4], "ReD": [69.2, 94.4, 94.4]},
    ("gsm8k", "8b", 0.25, 0.05): {"STC": [67.5, 90.3, 90.3], "ReD": [77.3, 90.3, 90.3]},
    ("mmlu-pro-500", "8b", 0.02, 0.1): {"STC": [12.9, 38.4, 63.7], "ReD": [41.8, 73.1, 83.5]},
    ("mmlu-pro-500", "8b", 0.08, 0.15): {"STC": [18.8, 56.1, 75.0], "ReD": [39.5, 70.4, 75.0]},
    ("mmlu-pro-500", "8b", 0.25, 0.05): {"STC": [29.5, 63.1, 63.1], "ReD": [44.2, 63.1, 63.1]},
}


def coverage_at(curves, budget):
    """Coverage (fraction of N) of each realization at a budget of `budget` attempts, interpolated linearly
    between the points of its curve (np.interp), as in the original table scripts (extract_table.py)."""
    return np.array([np.interp(budget, c["x"], c["y"]) for c in curves])


def run_job(job):
    dataset, model, fpr, fnr, realizations, seed = job
    random.seed(seed)
    path = os.path.join(REPO, "data", dataset, f"{MODELS[model]}_matrix.jsonl")
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(open(os.devnull, "w")):
        tasks, k = ss.load_matrix(path)
        stc, _ = ss.simulate_standard(tasks, realizations, metric="attempts", fpr=fpr, fnr=fnr)
        red, _ = ss.simulate_red(tasks, realizations, metric="attempts", fpr=fpr, fnr=fnr)
    n = len(tasks)
    res = {}
    for name, curves in (("STC", stc), ("ReD", red)):
        vals = [coverage_at(curves, m * n) for m in BUDGETS[dataset]]
        res[name] = [(100 * v.mean(), 100 * v.std(ddof=1) / np.sqrt(len(v))) for v in vals]
    print(f"done: {DATASETS[dataset]} {model} FPR={fpr} FNR={fnr} (N={n}, k={k})", flush=True)
    return (dataset, model, fpr, fnr), res


def fmt(cells):
    return " | ".join(f"{m:.1f} ± {s:.1f}" for m, s in cells)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--realizations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "tables", "tables_s1_s2_s4_s5_gsm8k_mmlu.md"))
    a = ap.parse_args()

    jobs = [(d, m, 0.0, 0.0) for d in DATASETS for m in MODELS]
    jobs += [(d, "8b", fpr, fnr) for d in DATASETS for fpr, fnr in NOISE]
    jobs = [j + (a.realizations, a.seed + i) for i, j in enumerate(jobs)]
    with mp.Pool(a.workers) as pool:
        results = dict(pool.map(run_job, jobs))

    L = []
    for t, d in (("1", "gsm8k"), ("2", "mmlu-pro-500")):
        hdr = " | ".join(f"{m} x N" for m in BUDGETS[d])
        L += [f"**Supplementary Table {t}. {DATASETS[d]}** (perfect verifier, {a.realizations} realizations; mean ± s.e., %)", "",
              f"| Model | Method | {hdr} |", "|---|---|" + "---|" * len(BUDGETS[d])]
        for m in MODELS:
            r = results[(d, m, 0.0, 0.0)]
            L += [f"| {m} | STC | {fmt(r['STC'])} |", f"| | ReD | {fmt(r['ReD'])} |"]
        L.append("")
    for t, d, nb in (("4", "gsm8k", 3), ("5", "mmlu-pro-500", 3)):
        bud = BUDGETS[d][:nb]
        hdr = " | ".join(f"{m} x N" for m in bud)
        L += [f"**Supplementary Table {t}. {DATASETS[d]}, llama-3.1-8b, imperfect verifier** "
              f"({a.realizations} realizations; true coverage, mean ± s.e., %)", "",
              f"| Verifier | Method | {hdr} |", "|---|---|" + "---|" * nb]
        for fpr, fnr in [(0.0, 0.0)] + NOISE:
            r = results[(d, "8b", fpr, fnr)]
            lab = "Perfect" if fpr == fnr == 0 else f"FPR={fpr}, FNR={fnr}"
            L += [f"| {lab} | STC | {fmt(r['STC'][:nb])} |", f"| | ReD | {fmt(r['ReD'][:nb])} |"]
        L.append("")

    # side-by-side check against the paper (the paper rounds to 0.1 percentage points)
    L += ["**Check against the paper** (this run minus paper, percentage points)", "",
          "| Case | Method | differences |", "|---|---|---|"]
    worst = 0.0
    for key, paper in PAPER.items():
        r = results[key]
        d, m, fpr, fnr = key
        for name in ("STC", "ReD"):
            diffs = [round(mean, 1) - p for (mean, se), p in zip(r[name], paper[name])]
            worst = max(worst, max(abs(x) for x in diffs))
            L.append(f"| {DATASETS[d]} {m} FPR={fpr} FNR={fnr} | {name} | "
                     + ", ".join(f"{x:+.1f}".replace("-0.0", "+0.0") for x in diffs) + " |")
    L.append("")
    L.append(f"Largest difference from the paper: {worst:.1f} percentage points.")
    md = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(md)
    print(md)
    print(f"[wrote {a.out}]")


if __name__ == "__main__":
    main()
