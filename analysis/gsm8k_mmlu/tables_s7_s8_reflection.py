"""Supplementary Tables 7 and 8: ReD compared with continuous reflection (CR) and with Reflection-ReD on GSM8K.

  STC, ReD   independent attempts (data/gsm8k/{model}_matrix.jsonl, 100 per problem), simulate_strategies.py
  CR         continuous reflection: the model gets its previous wrong answer back and tries again, up to 5 turns per
             chain; when a chain fails, a new independent chain starts (10 chains per problem, at most 50 attempts);
             problems are attempted one by one, as in STC (data/gsm8k_reflection/{model}_reflection_matrix.jsonl)
  Refl-ReD   the same chains, but scheduled as in ReD: one turn per unsolved problem per round, keeping each chain's
             context between rounds
CR and Refl-ReD are simulated by simulate_reflection.py (continuous_reflection branch of ReD4LLMs, unchanged).
Coverage at 1, 2 and 3 x N attempts, interpolated linearly on each realization's curve and averaged over 1,000
realizations, as in the original table script (extract_table.py).

The original runs were not seeded; the last digit can differ from the paper by about one standard error. The paper's
Tables 7 and 8 come from one such run, which is why ReD (8b) at 2 x N reads 98.2% in Table 8 but 98.1% in Tables 1
and 7, and ReD (70b) at 3 x N 97.4% vs 97.3%.

In the llama-3.1-8b reflection file, 2,597 attempts from an earlier generation run carry no chain_idx; the loader
assigns them to chain 0, as in the runs behind the paper.

Run (from anywhere; ~3 min on 2 cores):
    python analysis/gsm8k_mmlu/tables_s7_s8_reflection.py
Output: outputs/tables/tables_s7_s8_reflection.md
"""
import argparse
import contextlib
import multiprocessing as mp
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate_strategies as ss  # noqa: E402
import simulate_reflection as sr  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = {"8b": "llama-3.1-8b-instant", "70b": "llama-3.3-70b-versatile"}
MULT = [1, 2, 3]
N = 1319
PAPER = {  # Table 7 (STC, CR, ReD) and Table 8 (Refl-ReD, ReD)
    ("STC", "8b"): [33.6, 66.7, 98.3], ("CR", "8b"): [32.1, 63.6, 95.3], ("ReD", "8b"): [81.3, 98.1, 99.0],
    ("STC", "70b"): [25.1, 49.6, 74.4], ("CR", "70b"): [39.9, 78.9, 97.6], ("ReD", "70b"): [93.2, 97.0, 97.3],
    ("Refl-ReD", "8b"): [81.4, 94.2, 98.3], ("Refl-ReD", "70b"): [93.3, 97.5, 97.6],
}


def cov(curves, n):
    vals = [[np.interp(m * n, c["x"], c["y"]) for c in curves] for m in MULT]
    return [(100 * np.mean(v), 100 * np.std(v, ddof=1) / np.sqrt(len(v))) for v in vals]


def run_job(job):
    kind, model, R, seed = job
    random.seed(seed)
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(open(os.devnull, "w")):
        if kind == "independent":
            tasks, _ = ss.load_matrix(os.path.join(REPO, "data", "gsm8k", f"{MODELS[model]}_matrix.jsonl"))
            a, _ = ss.simulate_standard(tasks, R, metric="attempts")
            b, _ = ss.simulate_red(tasks, R, metric="attempts")
            out = {"STC": cov(a, len(tasks)), "ReD": cov(b, len(tasks))}
        else:
            tasks = sr.load_reflection_matrix(
                os.path.join(REPO, "data", "gsm8k_reflection", f"{MODELS[model]}_reflection_matrix.jsonl"))
            a = sr.simulate_continuous_reflection(tasks, R, metric="attempts")
            b = sr.simulate_reflection_red(tasks, R, metric="attempts")
            out = {"CR": cov(a, len(tasks)), "Refl-ReD": cov(b, len(tasks))}
    print(f"done: {kind} {model}", flush=True)
    return model, out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--realizations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "tables", "tables_s7_s8_reflection.md"))
    a = ap.parse_args()

    jobs = [(k, m, a.realizations, a.seed + i) for i, (k, m) in
            enumerate([(k, m) for k in ("independent", "reflection") for m in MODELS])]
    res = {}
    with mp.Pool(a.workers) as pool:
        for model, out in pool.map(run_job, jobs):
            for name, v in out.items():
                res[(name, model)] = v

    fmt = lambda cells: " | ".join(f"{m:.1f} ± {s:.1f}" for m, s in cells)  # noqa: E731
    L = [f"**Supplementary Table 7. ReD vs continuous reflection (CR), GSM8K** (N={N}, {a.realizations} "
         "realizations; coverage, mean ± s.e., %)", "", "| Strategy | 1 x N | 2 x N | 3 x N |", "|---|---|---|---|"]
    for m in MODELS:
        for name in ("STC", "CR", "ReD"):
            L.append(f"| {name} ({m}) | {fmt(res[(name, m)])} |")
    L += ["", f"**Supplementary Table 8. ReD beyond independent repeated sampling, GSM8K** ({a.realizations} "
          "realizations; coverage, mean ± s.e., %)", "", "| Strategy | 1 x N | 2 x N | 3 x N |", "|---|---|---|---|"]
    for m in MODELS:
        for name in ("Refl-ReD", "ReD"):
            L.append(f"| {name} ({m}) | {fmt(res[(name, m)])} |")
    L += ["", "**Check against the paper** (this run minus paper, percentage points; ReD compared with Table 7)", "",
          "| Strategy | differences |", "|---|---|"]
    worst = 0.0
    for (name, m), p in PAPER.items():
        d = [round(v[0], 1) - x for v, x in zip(res[(name, m)], p)]
        worst = max(worst, max(abs(x) for x in d))
        L.append(f"| {name} ({m}) | " + ", ".join(f"{x:+.1f}".replace("-0.0", "+0.0") for x in d) + " |")
    L += ["", f"Largest difference from the paper: {worst:.1f} percentage points."]
    md = "\n".join(L) + "\n"
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        f.write(md)
    print(md)
    print(f"[wrote {a.out}]")


if __name__ == "__main__":
    main()
