"""Vectorized simulation of solve-to-completion (STC) and ReD (tau = 1) on a shuffled N x k attempt matrix, counting
the budget in attempts and in output tokens. Used by table_s11_reasoning_effort_mmlu.py (Supplementary Table 11).

Copied unchanged from coverage_simulation.py of the ReD4LLMs repository.
"""
import numpy as np


def _stc_trajectory(C_shuf, O_shuf):
    """
    Vectorized Solve-to-Completion trajectory for one shuffled (C, O) pair.
    Rows/columns of C_shuf, O_shuf are already permuted for this realization.
    One waypoint per question (row-boundary resolution) -- matches this repo's
    existing simulate_strategies.py convention of recording a point per row
    resolution rather than per individual sub-attempt within a row.
    """
    N, K = C_shuf.shape
    has_success = C_shuf.any(axis=1)
    first_idx = np.argmax(C_shuf, axis=1)  # 0 when no True; masked out via has_success below
    attempts_per_row = np.where(has_success, first_idx + 1, K)

    O_cumsum = np.cumsum(O_shuf, axis=1)
    tokens_per_row = np.take_along_axis(O_cumsum, (attempts_per_row - 1)[:, None], axis=1).squeeze(1)

    cum_attempts = np.concatenate([[0], np.cumsum(attempts_per_row)])
    cum_tokens = np.concatenate([[0], np.cumsum(tokens_per_row)])
    cum_coverage = np.concatenate([[0], np.cumsum(has_success.astype(np.int64))])
    return cum_attempts, cum_tokens, cum_coverage


def _red_trajectory(C_shuf, O_shuf):
    """
    Round-robin Reset-and-Discard (tau=1) trajectory for one shuffled (C, O)
    pair. Rows are already in shuffled row order; round j processes the
    currently-unsolved rows in that same fixed order, spending exactly one
    attempt each on column j, and records a waypoint at every success (so
    successes within the same round are NOT bunched into one round-end point --
    each gets its own, cheaper, within-round cumulative cost).
    """
    N, K = C_shuf.shape
    solved = np.zeros(N, dtype=bool)
    prior_attempts = 0
    prior_tokens = 0.0
    prior_coverage = 0

    wp_attempts = [0]
    wp_tokens = [0.0]
    wp_coverage = [0]

    for j in range(K):
        if solved.all():
            break
        unsolved_idx = np.nonzero(~solved)[0]
        tokens_this_round = O_shuf[unsolved_idx, j]
        success_this_round = C_shuf[unsolved_idx, j]

        within_cum_attempts = np.arange(1, unsolved_idx.size + 1)
        within_cum_tokens = np.cumsum(tokens_this_round)

        succ_positions = np.nonzero(success_this_round)[0]
        if succ_positions.size > 0:
            wp_attempts.extend((prior_attempts + within_cum_attempts[succ_positions]).tolist())
            wp_tokens.extend((prior_tokens + within_cum_tokens[succ_positions]).tolist())
            wp_coverage.extend((prior_coverage + np.arange(1, succ_positions.size + 1)).tolist())

        solved[unsolved_idx[success_this_round]] = True
        prior_attempts += int(unsolved_idx.size)
        prior_tokens += float(tokens_this_round.sum())
        prior_coverage += int(success_this_round.sum())

    if wp_attempts[-1] != prior_attempts:
        wp_attempts.append(prior_attempts)
        wp_tokens.append(prior_tokens)
        wp_coverage.append(prior_coverage)

    return np.array(wp_attempts, dtype=np.float64), np.array(wp_tokens, dtype=np.float64), np.array(wp_coverage, dtype=np.float64)


def _crossing_stats(cum_x, cum_coverage, targets):
    """First cum_x value at which cum_coverage >= target, for each target. NaN if never reached."""
    out = []
    for t in targets:
        idx = np.searchsorted(cum_coverage, t, side="left")
        out.append(float(cum_x[idx]) if idx < len(cum_coverage) else float("nan"))
    return out


def run_monte_carlo(C, O, R=500, seed=0, coverage_targets_frac=(0.25, 0.5, 0.75), grid_points=1500):
    """
    C: (N,K) bool correctness matrix. O: (N,K) numeric output-token matrix.
    Returns mean coverage/N curves vs attempts and vs tokens for both STC and
    ReD(tau=1), plus mean attempts/tokens to first reach each coverage target
    (averaged over R realizations), each strategy simulated on the SAME
    per-realization row+column shuffle for a matched comparison.
    """
    N, K = C.shape
    rng = np.random.default_rng(seed)
    targets = [f * N for f in coverage_targets_frac]

    stc_trajs, red_trajs = [], []

    for _ in range(R):
        perm_rows = rng.permutation(N)
        perm_cols = rng.permutation(K)
        C_shuf = C[perm_rows][:, perm_cols]
        O_shuf = O[perm_rows][:, perm_cols]

        stc = _stc_trajectory(C_shuf, O_shuf)
        red = _red_trajectory(C_shuf, O_shuf)

        # Exact invariant (not just on average): both strategies exhaust the
        # full K-attempt budget for any question that's never solved, so final
        # coverage must equal raw pass@K exactly, every single realization.
        assert stc[2][-1] == red[2][-1], "STC/ReD final coverage mismatch -- simulation bug"

        stc_trajs.append(stc)
        red_trajs.append(red)

    true_pass_at_k_count = int(C.any(axis=1).sum())
    assert int(stc_trajs[0][2][-1]) == true_pass_at_k_count, "Final coverage != raw pass@K -- simulation bug"

    def aggregate(trajs, x_index):
        max_x = max(traj[x_index][-1] for traj in trajs)
        grid = np.linspace(0, max_x, grid_points)
        y_sum = np.zeros(grid_points)
        for traj in trajs:
            y_sum += np.interp(grid, traj[x_index], traj[2])
        return grid, y_sum / len(trajs)

    def crossing_summary(trajs, x_index):
        per_target = [[] for _ in targets]
        for traj in trajs:
            vals = _crossing_stats(traj[x_index], traj[2], targets)
            for ti, v in enumerate(vals):
                per_target[ti].append(v)
        # nan when a target coverage level is never reached in any realization
        # (e.g. requesting 75% when true pass@K < 75%) -- not a warning-worthy case.
        mean_vals = [float(np.nanmean(v)) if not np.all(np.isnan(v)) else float("nan") for v in per_target]
        n_unreached = [int(np.sum(np.isnan(v))) for v in per_target]
        return mean_vals, n_unreached

    def summarize(trajs):
        attempts_grid, coverage_by_attempts = aggregate(trajs, 0)
        tokens_grid, coverage_by_tokens = aggregate(trajs, 1)
        cross_attempts, unreached_attempts = crossing_summary(trajs, 0)
        cross_tokens, unreached_tokens = crossing_summary(trajs, 1)
        return {
            "attempts_grid": attempts_grid, "coverage_by_attempts": coverage_by_attempts,
            "tokens_grid": tokens_grid, "coverage_by_tokens": coverage_by_tokens,
            "targets_frac": list(coverage_targets_frac),
            "cross_attempts": cross_attempts, "cross_tokens": cross_tokens,
            "n_unreached_attempts": unreached_attempts, "n_unreached_tokens": unreached_tokens,
        }

    return {
        "N": N, "K": K, "R": R,
        "true_pass_at_k": true_pass_at_k_count / N,
        "stc": summarize(stc_trajs),
        "red": summarize(red_trajs),
    }
