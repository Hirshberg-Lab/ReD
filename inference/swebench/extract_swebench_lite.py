"""Build data/swebench/swebench_lite_deepseek_samples.csv from the SWE-bench Lite samples of Brown et al.

Source (public GitHub repository of Brown et al., 2024): https://github.com/ScalingIntelligence/swe-bench-lite-samples
  summary.json                      per-sample correctness ("resolved") for the 300 issues, split into
                                    "instances_without_flaky_tests" (266) and "instances_with_flaky_tests" (34)
  trajectories/<issue>/<j>.json     the Moatless Tools trajectory of sample j (DeepSeek-Coder-V2-Instruct,
                                    temperature 1.6); every model call records input_tokens, output_tokens and
                                    completion_cost (USD at $3 / $15 per million input / output tokens)

We keep the 266 issues without flaky tests, so that every sample is checked by a single run of the unit tests.
Samples that crashed are absent from summary.json (15 issues have 248-249 samples); the simulation counts the
missing ones as failures.

Get the source (about 330 MB download, 2.5 GB on disk for the 266 issues):
    git clone --filter=blob:none --no-checkout https://github.com/ScalingIntelligence/swe-bench-lite-samples
    cd swe-bench-lite-samples && git sparse-checkout set --no-cone /summary.json /trajectories/ && git checkout main
Then:
    python inference/swebench/extract_swebench_lite.py --src path/to/swe-bench-lite-samples
Output columns: issue, sample, resolved, n_steps (agent steps = Moatless transitions; mean 9.9),
                n_calls (model calls), input_tokens, output_tokens, usd
"""
import argparse
import csv
import json
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", required=True, help="clone of ScalingIntelligence/swe-bench-lite-samples")
    ap.add_argument("--out", default=os.path.join(REPO, "data", "swebench", "swebench_lite_deepseek_samples.csv"))
    a = ap.parse_args()

    summary = json.load(open(os.path.join(a.src, "summary.json")))["instances_without_flaky_tests"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    n_rows = 0
    with open(a.out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["issue", "sample", "resolved", "n_steps", "n_calls", "input_tokens", "output_tokens", "usd"])
        for issue in sorted(summary):
            for j in sorted(summary[issue], key=int):
                traj = json.load(open(os.path.join(a.src, "trajectories", issue, f"{j}.json")))
                calls = [act for tr in traj["transitions"] for act in tr["actions"]]
                inp = sum(act.get("input_tokens") or 0 for act in calls)
                out = sum(act.get("output_tokens") or 0 for act in calls)
                usd = sum(act.get("completion_cost") or 0.0 for act in calls)
                w.writerow([issue, j, int(bool(summary[issue][j]["resolved"])), len(traj["transitions"]), len(calls),
                            inp, out, f"{usd:.6f}"])
                n_rows += 1
    print(f"wrote {n_rows} samples of {len(summary)} issues to {a.out}")


if __name__ == "__main__":
    main()
