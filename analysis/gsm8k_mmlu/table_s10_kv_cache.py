"""Supplementary Table 10: coverage on GSM8K at fixed GPU time, when STC keeps the KV cache and ReD does not.

GPU time of one attempt: N_in * L_p (prefill) + N_out * L_d (decode), with L_p = 5 ms/token and L_d = 20 ms/token and
the measured input and output tokens of each attempt (data/gsm8k/{model}_matrix.jsonl). STC pays the prefill once per
problem (KV cache kept between its attempts); strict ReD pays it on every attempt. The simulator is
simulate_hardware.py (KV_cache branch of ReD4LLMs, unchanged). Coverage at 2,219, 11,516 and 21,952 GPU-seconds
(the budgets of the original table script, verify_table.py), interpolated linearly on each realization's curve and
averaged over 1,000 realizations.

The original runs were not seeded; the last digit can differ from the paper by about one standard error.

Run (from anywhere; ~3 min on 2 cores):
    python analysis/gsm8k_mmlu/table_s10_kv_cache.py
Output: outputs/tables/table_s10_kv_cache.md
"""
import argparse
import contextlib
import multiprocessing as mp
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import simulate_hardware as sh  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS = {"8b": "llama-3.1-8b-instant", "70b": "llama-3.3-70b-versatile"}
GPU_S = [2219, 11516, 21952]
PREFILL_MS, DECODE_MS = 5.0, 20.0
PAPER = {("STC", "8b"): [10.1, 50.0, 94.5], ("ReD", "8b"): [26.0, 95.6, 98.8],
         ("STC", "70b"): [9.5, 47.2, 89.6], ("ReD", "70b"): [34.0, 96.9, 97.4]}


def cov(curves):
    vals = [[np.interp(g, c["x"], c["y"]) for c in curves] for g in GPU_S]
    return [(100 * np.mean(v), 100 * np.std(v, ddof=1) / np.sqrt(len(v))) for v in vals]


def run_job(job):
    model, R, seed = job
    random.seed(seed)
    with contextlib.redirect_stdout(open(os.devnull, "w")), contextlib.redirect_stderr(open(os.devnull, "w")):
        tasks, _ = sh.load_matrix(os.path.join(REPO, "data", "gsm8k", f"{MODELS[model]}_matrix.jsonl"))
        stc, _ = sh.simulate_hardware_standard(tasks, R, PREFILL_MS, DECODE_MS)
        red, _ = sh.simulate_hardware_strict_red(tasks, R, PREFILL_MS, DECODE_MS)
    print(f"done: {model}", flush=True)
    return model, {"STC": cov(stc), "ReD": cov(red)}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--realizations", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--out", default=os.path.join(REPO, "outputs", "tables", "table_s10_kv_cache.md"))
    a = ap.parse_args()

    with mp.Pool(a.workers) as pool:
        res = dict(pool.map(run_job, [(m, a.realizations, a.seed + i) for i, m in enumerate(MODELS)]))

    L = [f"**Supplementary Table 10. GPU time with KV cache for STC only, GSM8K** (N=1,319, {a.realizations} "
         f"realizations, L_p = {PREFILL_MS:g} ms/token, L_d = {DECODE_MS:g} ms/token; coverage, mean ± s.e., %)", "",
         "| GPU-s | Method | 8b | 70b |", "|---|---|---|---|"]
    for gi, g in enumerate(GPU_S):
        for name in ("STC", "ReD"):
            cells = " | ".join(f"{res[m][name][gi][0]:.1f} ± {res[m][name][gi][1]:.1f}" for m in MODELS)
            L.append(f"| {g:,} | {name} | {cells} |" if name == "STC" else f"| | {name} | {cells} |")
    L += ["", "**Check against the paper** (this run minus paper, percentage points)", "",
          "| Method | differences at 2,219 / 11,516 / 21,952 GPU-s |", "|---|---|"]
    worst = 0.0
    for (name, m), p in PAPER.items():
        d = [round(res[m][name][gi][0], 1) - x for gi, x in enumerate(p)]
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
