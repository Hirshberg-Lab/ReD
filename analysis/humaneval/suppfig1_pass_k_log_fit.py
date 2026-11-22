"""Supplementary Fig. 1: power-law fit of 1 - pass@k for llama-3.1-8b-instant on HumanEval.

pass@k is computed with the unbiased estimator of Chen et al. (2021) from the 164 x 100
correctness matrix; log(1 - pass@k) is fitted linearly against log k for k >= 10.
The fitted slope is -alpha; the legend of the figure shows "exponent = -0.336", i.e. alpha = 0.34.

Run (from anywhere):
    python analysis/humaneval/suppfig1_pass_k_log_fit.py
Input:   data/humaneval/llama-3.1-8b-instant_correctness.csv
Outputs: outputs/figures/llama-3.1-8b-instant_pass_k_log_fit.png (Supplementary Fig. 1)
         outputs/figures/llama-3.1-8b-instant_pass_k.png (pass@k, linear axes; not in the paper)
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_coverage import read_correctness, plot_pass_k  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--model", default="llama-3.1-8b-instant")
    ap.add_argument("--results_dir", default=os.path.join(REPO, "data", "humaneval"))
    ap.add_argument("--output_dir", default=os.path.join(REPO, "outputs", "figures"))
    a = ap.parse_args()
    os.makedirs(a.output_dir, exist_ok=True)
    data = read_correctness(os.path.join(a.results_dir, f"{a.model}_correctness.csv"))
    # plot_pass_k writes {model}_pass_k.png and, next to it, {model}_pass_k_log_fit.png
    plot_pass_k(data, os.path.join(a.output_dir, f"{a.model}_pass_k.png"), "Pass@k", a.model)


if __name__ == "__main__":
    main()
