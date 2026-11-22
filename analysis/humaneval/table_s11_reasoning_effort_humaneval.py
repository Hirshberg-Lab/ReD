"""Supplementary Table 11, HumanEval rows: saving of ReD for gpt-oss-20b at low and medium reasoning effort.

Saving = (budget that solve-to-completion (STC) needs to reach a target coverage) / (budget of ReD),
for targets of 25%, 50% and 75% of N = 164, with the budget counted in attempts or in output tokens.
The budget to reach a target is averaged over 200 realizations (random shuffles of the order of the
problems and of the attempts of each problem), seed 0.

Medium effort is the run used in the main text (data/humaneval/openai_gpt-oss-20b_*);
low effort is a separate run with the same prompt and settings (data/humaneval/openai_gpt-oss-20b_low_*).
The simulators are those of plot_coverage.py (STC: simulate_realizations; ReD with tau = 1:
simulate_red_realizations). Adapted from the script that produced the paper's numbers,
keeping only the table that is in the paper.

Run (from anywhere):
    python analysis/humaneval/table_s11_reasoning_effort_humaneval.py
Output: outputs/tables/table_s11_reasoning_effort_humaneval.md
"""
import os
import sys
import random
import contextlib
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_coverage import (  # noqa: E402
    read_full_data,
    simulate_realizations,       # STC
    simulate_red_realizations,   # ReD, tau=1
)

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(REPO, "data", "humaneval")
OUT_MD = os.path.join(REPO, "outputs", "tables", "table_s11_reasoning_effort_humaneval.md")

N = 164
K = 100
R = 200          # realizations
SEED = 0
TARGETS = [0.25, 0.50, 0.75]
EFFORTS = ["low", "medium"]

SOURCES = {
    "low":    "openai_gpt-oss-20b_low",
    "medium": "openai_gpt-oss-20b",
}


def load_arm(effort):
    """Load one arm's matrices, pad every problem to exactly K samples (a missing
    (problem, attempt) cell is padded as a FAILURE with 0 tokens), and return the
    padded tasks dict plus the count of padded/missing cells."""
    prefix = SOURCES[effort]
    cfile = os.path.join(DATA, f"{prefix}_correctness.csv")
    tfile = os.path.join(DATA, f"{prefix}_tokens.csv")
    for f in (cfile, tfile):
        if not os.path.exists(f):
            sys.exit(f"STOP: missing file for {effort}: {f}")

    tasks = read_full_data(cfile, tfile)
    if len(tasks) != N:
        sys.exit(f"STOP: {effort} has {len(tasks)} problems, expected N={N}.")

    missing = 0
    for tid, samples in tasks.items():
        # ensure every present sample has a tokens field
        for s in samples:
            s.setdefault("tokens", 0)
        while len(samples) < K:
            samples.append({"passed": False, "tokens": 0})  # missing -> failure
            missing += 1
        if len(samples) > K:
            sys.exit(f"STOP: {effort} problem {tid} has {len(samples)} > K={K} samples.")
    return tasks, missing


def budget_to_target(sims, target):
    """Mean over realizations of the cumulative cost (x) at which coverage (y)
    first reaches `target`. Errors out if a realization never reaches the target."""
    xs = []
    for x, y in sims:
        x = np.asarray(x, float); y = np.asarray(y, float)
        idx = int(np.argmax(y >= target))
        if y[idx] < target:
            raise RuntimeError(f"target {target} never reached in a realization "
                               f"(max coverage {y.max():.3f})")
        xs.append(x[idx])
    return float(np.mean(xs))


def sim_all(tasks):
    """Run STC and ReD, for both attempts and output tokens, R realizations each,
    seeded for reproducibility (the simulators print progress; silenced here)."""
    out = {}
    plan = [("stc", "attempts", simulate_realizations),
            ("red", "attempts", simulate_red_realizations),
            ("stc", "tokens",   simulate_realizations),
            ("red", "tokens",   simulate_red_realizations)]
    for policy, metric, fn in plan:
        random.seed(SEED)  # identical seed -> att and tok share trajectories per policy
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            out[(policy, metric)] = fn(tasks, R, metric)
    return out


def main():
    lines = ["**Supplementary Table 11, HumanEval rows.** Saving of ReD (STC budget / ReD budget) "
             f"for gpt-oss-20b, N={N}, {R} realizations.", "",
             "| Effort | Attempts 25% | Attempts 50% | Attempts 75% | Output tokens 25% | Output tokens 50% | Output tokens 75% |",
             "|---|---|---|---|---|---|---|"]
    for effort in EFFORTS:
        tasks, missing = load_arm(effort)
        print(f"{effort}: {missing} missing (problem, attempt) cells, counted as failures")
        s = sim_all(tasks)
        att = [budget_to_target(s[("stc", "attempts")], f) / budget_to_target(s[("red", "attempts")], f) for f in TARGETS]
        tok = [budget_to_target(s[("stc", "tokens")], f) / budget_to_target(s[("red", "tokens")], f) for f in TARGETS]
        lines.append(f"| {effort} | " + " | ".join(f"{v:.1f}" for v in att + tok) + " |")
    md = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w") as f:
        f.write(md)
    print(md)
    print(f"[wrote {OUT_MD}]")


if __name__ == "__main__":
    main()
