"""Library functions used by plot_figures.py (Figs. 4-6).

Trimmed copy of plotting/plot_paper_figures.py from the ReD repository: only the functions that the
paper's figures and tables call are kept, with their code unchanged.
"""
import csv
import os
import random
import numpy as np
import math

def read_full_data(correctness_file, tokens_file):
    """
    Reads both CSVs and returns a dictionary: task_id -> list of {'passed': bool, 'tokens': int}
    """
    tasks = {}
    if not os.path.exists(correctness_file) or not os.path.exists(tokens_file):
        print(f"Warning: Files not found: {correctness_file} or {tokens_file}")
        return {}
        
    # Read Correctness
    with open(correctness_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row: continue
            tid = row[0]
            val_list = []
            for val in row[1:]:
                if val.strip():
                    val_list.append(int(val) == 1)
            tasks[tid] = [{'passed': v} for v in val_list]
            
    # Read Tokens and merge
    with open(tokens_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            if not row: continue
            tid = row[0]
            if tid in tasks:
                vals = [int(v) for v in row[1:] if v.strip()]
                for i, toks in enumerate(vals):
                    if i < len(tasks[tid]):
                        tasks[tid][i]['tokens'] = toks
    return tasks


def calculate_curve(tasks_list, metric="attempts"):
    total_problems = len(tasks_list)
    cumulative_cost = 0
    solved_count = 0
    x = [0]
    y = [0.0]
    for samples in tasks_list:
        solved_this_task = False
        for s in samples:
            cost = 1 if metric == "attempts" else s.get('tokens', 0)
            cumulative_cost += cost
            if s['passed']:
                solved_this_task = True
                solved_count += 1
                x.append(cumulative_cost)
                y.append(solved_count / total_problems)
                break
        if not solved_this_task:
            x.append(cumulative_cost)
            y.append(solved_count / total_problems)
    return x, y


def simulate_realizations(tasks_dict, n_realizations, metric):
    realization_curves = []
    task_keys = list(tasks_dict.keys())
    for _ in range(n_realizations):
        shuffled_keys = task_keys[:]
        random.shuffle(shuffled_keys)
        current_task_list = []
        for tid in shuffled_keys:
            samples = tasks_dict[tid][:]
            random.shuffle(samples)
            current_task_list.append(samples)
        rx, ry = calculate_curve(current_task_list, metric)
        realization_curves.append((rx, ry))
    return realization_curves


def simulate_red_realizations(tasks_dict, n_realizations, metric):
    realization_curves = []
    task_keys = list(tasks_dict.keys())
    total_problems = len(task_keys)
    
    for _ in range(n_realizations):
        task_samples = {}
        for tid in task_keys:
            samps = tasks_dict[tid][:]
            random.shuffle(samps)
            task_samples[tid] = samps
            
        unsolved_ids = task_keys[:]
        task_indices = {tid: 0 for tid in task_keys}
        cumulative_cost = 0
        solved_count = 0
        x = [0]
        y = [0.0]
        
        while unsolved_ids:
            random.shuffle(unsolved_ids)
            next_round_unsolved = []
            for tid in unsolved_ids:
                idx = task_indices[tid]
                samples = task_samples[tid]
                if idx >= len(samples): continue
                
                s = samples[idx]
                task_indices[tid] += 1
                cost = 1 if metric == "attempts" else s.get('tokens', 0)
                cumulative_cost += cost
                
                if s['passed']:
                    solved_count += 1
                    x.append(cumulative_cost)
                    y.append(solved_count / total_problems)
                else:
                    if task_indices[tid] < len(samples):
                        next_round_unsolved.append(tid)
                    else:
                        x.append(cumulative_cost)
                        y.append(solved_count / total_problems)
            unsolved_ids = next_round_unsolved
        realization_curves.append((x, y))
    return realization_curves


def calculate_red_Rn(data, num_realizations=100):
    """
    Simulates ReD and returns the average sequence of active problems R_n.
    R_n: Number of problems attempted in round n (i.e., unsolved at start of round n).
    """
    all_Rn_sequences = []
    
    # Pre-shuffle data for realizations
    problem_ids = list(data.keys())
    
    for _ in range(num_realizations):
        # State: active problems
        active_problems = list(problem_ids) # All measurable problems start active
        rn_sequence = []
        
        # Shuffle samples for this realization to get different trajectories
        realization_data = {}
        for pid in problem_ids:
            samples = list(data[pid])
            random.shuffle(samples)
            realization_data[pid] = samples
            
        # We need to track how many attempts we've used per problem to index into samples
        # But samples are just a list. We can just pop or track index.
        # Since we shuffle/randomize, we can just use index 0, 1, 2...
        problem_indices = {pid: 0 for pid in problem_ids}
        
        # Simulation loop
        # Round 1, 2, ...
        # Continues until no active problems or we run out of samples for all
        
        round_num = 1
        while active_problems:
            rn_sequence.append(len(active_problems))
            
            next_active = []
            for pid in active_problems:
                # Check if we have a sample for this round
                idx = problem_indices[pid]
                samples = realization_data[pid]
                
                if idx < len(samples):
                    # We have a sample
                    is_solved = samples[idx]['passed']
                    problem_indices[pid] += 1
                    
                    if not is_solved:
                        next_active.append(pid)
                    # else: solved, removed from active
                else:
                    # No more samples, effectively "gave up" or treated as unsolved forever
                    # For scaling law, we usually assume infinite budget or cut off?
                    # If we run out of samples, R_n might artificially drop or stay constant?
                    # Standard ReD simulation usually has a fixed budget or max tokens.
                    # Here we simulate until max samples of data?
                    # If we exclude it, R_n drops. That might look like a solve.
                    # Safest: Keep it active but can't solve? Or drop?
                    # If we keep it in active but never increment index, it stays active forever.
                    # This flattens R_n -> R_{n+1}-R_n goes to 0 -> y_n blows up.
                    # The formula relies on solves.
                    next_active.append(pid)
                    
            active_problems = next_active
            
            # Break if static (no samples left for any active problem)
            # If all active problems have exhausted samples
            all_exhausted = True
            for pid in active_problems:
                if problem_indices[pid] < len(data[pid]):
                    all_exhausted = False
                    break
            if all_exhausted:
                # Keep recording R_n? No, stop.
                break
                
            round_num += 1
            
            # Safety break
            if round_num > 1000: break
            
        all_Rn_sequences.append(rn_sequence)
        
    # Average R_n across realizations
    # Sequences might be different lengths
    max_len = max(len(s) for s in all_Rn_sequences)
    avg_Rn = []
    std_Rn = []
    
    for i in range(max_len):
        vals = [s[i] for s in all_Rn_sequences if i < len(s)]
        if not vals: break
        avg_Rn.append(np.mean(vals))
        std_Rn.append(np.std(vals, ddof=1) if len(vals) > 1 else 0.0)
        
    return avg_Rn, std_Rn


def estimate_pass_at_k(n, c, k):
    """
    Unbiased estimator for pass@k.
    """
    if n < k: return 0.0
    if c == n: return 1.0
    try:
        prob_fail = math.comb(n - c, k) / math.comb(n, k)
        return 1.0 - prob_fail
    except ValueError:
        return 0.0


def calculate_theoretical_red_curve(data):
    # Number of problems
    N = len(data)
    
    # Determine max attempts
    max_samples = 0
    problem_stats = [] # list of (n, c)
    for tid, samples in data.items():
        n = len(samples)
        if n > max_samples: max_samples = n
        c = sum(1 for s in samples if s['passed'])
        problem_stats.append((n, c))
        
    pass_at_k = [] 
    
    # Calculate global Pass@k by averaging over valid tasks
    for k in range(1, max_samples + 1):
        total_p = 0.0
        count = 0
        for n, c in problem_stats:
            if n >= k:
                total_p += estimate_pass_at_k(n, c, k)
                count += 1
        
        if count > 0:
            pass_at_k.append(total_p / count)
        else:
            pass_at_k.append(0.0) 
            
    # t(n) = N * sum_{k=0}^{n-1} (1 - pass@k)
    x_theory = []
    y_theory = []
    
    current_x = 0.0
    pass_at_previous = 0.0 # pass@0
    
    for k_idx, p_at_k in enumerate(pass_at_k):
        # term = N * (1 - pass@(k-1))
        term = N * (1.0 - pass_at_previous)
        current_x += term
        
        x_theory.append(current_x)
        y_theory.append(p_at_k)
        
        pass_at_previous = p_at_k
        
    return x_theory, y_theory
