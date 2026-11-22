import os
import json
import argparse
import numpy as np

from generate_matrix import UNK_SENTINEL, load_dataset_by_name


def build_matrices(jsonl_path, n_questions, k_samples, dataset_name="mmlu-pro-500"):
    correct = np.zeros((n_questions, k_samples), dtype=bool)
    out_tokens = np.zeros((n_questions, k_samples), dtype=np.int64)
    in_tokens = np.zeros((n_questions, k_samples), dtype=np.int64)
    predicted = np.full((n_questions, k_samples), UNK_SENTINEL, dtype="<U8")
    gold = np.array([""] * n_questions, dtype="<U8")
    seen = np.zeros((n_questions, k_samples), dtype=bool)

    n_lines = 0
    n_malformed_json = 0
    n_out_of_range = 0
    n_duplicate = 0

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n_lines += 1
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                # Can happen if a run was killed mid-write; never fabricate a cell for it.
                n_malformed_json += 1
                continue

            if row.get("error") is not None:
                continue  # exhausted-retry failures were never meant to be persisted; skip defensively

            try:
                i = int(row["task_id"])
                j = int(row["attempt_idx"])
            except (KeyError, ValueError, TypeError):
                n_malformed_json += 1
                continue

            if not (0 <= i < n_questions) or not (0 <= j < k_samples):
                n_out_of_range += 1
                continue

            if seen[i, j]:
                n_duplicate += 1  # keep the latest occurrence (falls through and overwrites)

            seen[i, j] = True
            correct[i, j] = bool(row.get("passed", False))
            out_tokens[i, j] = int(row.get("out_tokens", 0) or 0)
            in_tokens[i, j] = int(row.get("in_tokens", 0) or 0)

            row_gold = row.get("gold")
            row_pred = row.get("predicted")
            if row_gold is not None:
                gold[i] = row_gold
            if row_pred is not None:
                predicted[i, j] = row_pred

    n_missing = int((~seen).sum())
    n_unk_predicted_among_present = int(((predicted == UNK_SENTINEL) & seen).sum())

    # Backfill gold for any question with zero logged attempts (so the (N,)
    # gold vector is fully populated even where "seen" never fired), and for
    # older matrix files that predate the predicted/gold logging fields.
    empty_gold_rows = np.where(gold == "")[0]
    if len(empty_gold_rows) > 0:
        problems = load_dataset_by_name(dataset_name)
        for i in empty_gold_rows:
            entry = problems.get(str(i))
            if entry is not None:
                gold[i] = entry["exact_answer"]

    stats = {
        "jsonl_path": jsonl_path,
        "n_lines_read": n_lines,
        "n_malformed_json": n_malformed_json,
        "n_out_of_range": n_out_of_range,
        "n_duplicate_cells": n_duplicate,
        "n_cells_total": n_questions * k_samples,
        "n_cells_present": int(seen.sum()),
        "n_cells_missing": n_missing,
        "n_unk_predicted_among_present": n_unk_predicted_among_present,
    }

    return {
        "correct": correct,
        "out_tokens": out_tokens,
        "in_tokens": in_tokens,
        "predicted": predicted,
        "gold": gold,
        "seen": seen,
    }, stats


def consistency_check(mats):
    """
    (predicted == gold) must equal the stored correctness bit for every cell we
    actually have data for. This is tautological given how process_attempt logs
    both fields from the same extractor call -- a mismatch here means a bug in
    matrix construction (e.g. a row/column indexing error), not in the model.
    """
    predicted = mats["predicted"]
    gold = mats["gold"][:, None]  # (N,1) broadcasts against (N,K)
    correct = mats["correct"]
    seen = mats["seen"]

    derived_correct = (predicted == gold) & (predicted != UNK_SENTINEL)
    mismatch = (derived_correct != correct) & seen
    n_mismatch = int(mismatch.sum())
    locations = list(zip(*np.where(mismatch)))
    return n_mismatch, locations[:20]


def main():
    parser = argparse.ArgumentParser(description="Build dense N x K matrices from a generate_matrix.py JSONL, with a predicted/gold consistency check.")
    parser.add_argument("--matrix", type=str, required=True)
    parser.add_argument("--n_questions", type=int, default=500)
    parser.add_argument("--k_samples", type=int, required=True)
    parser.add_argument("--dataset", type=str, default="mmlu-pro-500")
    parser.add_argument("--out", type=str, default=None, help="Output .npz path (default: alongside input, same basename)")
    args = parser.parse_args()

    mats, stats = build_matrices(args.matrix, args.n_questions, args.k_samples, args.dataset)

    print("=== Matrix build stats ===")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    n_mismatch, mismatch_locations = consistency_check(mats)
    print("\n=== Consistency check: (predicted == gold) vs stored correctness bit ===")
    print(f"  mismatches: {n_mismatch} / {stats['n_cells_present']} present cells")
    if n_mismatch > 0:
        print(f"  first mismatch locations (row=question_idx, col=attempt_idx): {mismatch_locations}")
        print("  ^^ THIS INDICATES A BUG in matrix construction or logging -- investigate before trusting downstream results.")

    out_path = args.out or (os.path.splitext(args.matrix)[0] + "_matrices.npz")
    np.savez_compressed(
        out_path,
        correct=mats["correct"],
        out_tokens=mats["out_tokens"],
        in_tokens=mats["in_tokens"],
        predicted=mats["predicted"],
        gold=mats["gold"],
        seen=mats["seen"],
    )
    print(f"\nSaved matrices to {out_path}")

    # Honest accounting: missing cells are already False in `correct`, so they
    # count as failures here rather than being silently dropped from the mean.
    # This denominator is only meaningful once the sweep is complete (or the
    # residual missing count above is small) -- on a partial run it conflates
    # coverage with accuracy.
    pass_at_1 = mats["correct"].mean()
    pass_at_k = mats["correct"].any(axis=1).mean()
    pass_at_1_attempted = mats["correct"].sum() / max(stats["n_cells_present"], 1)
    print(f"\n=== pass@1 / pass@{args.k_samples} (missing cells = {stats['n_cells_missing']}, counted as failures) ===")
    print(f"pass@1 = {pass_at_1:.4f}  (of full {stats['n_cells_total']}-cell target)")
    print(f"pass@1 among attempted cells only = {pass_at_1_attempted:.4f}  (of {stats['n_cells_present']} present cells -- informational, NOT the headline number)")
    print(f"pass@{args.k_samples} = {pass_at_k:.4f}")


if __name__ == "__main__":
    main()
