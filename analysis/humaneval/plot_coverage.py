"""Library functions used by suppfig1_pass_k_log_fit.py (Supplementary Fig. 1) and table_s11_reasoning_effort_humaneval.py (Supplementary Table 11, HumanEval rows).

Trimmed copy of plot_coverage.py from the ReD repository: only the functions that the
paper's figures and tables call are kept, with their code unchanged.
"""
import os
import csv
import math
import numpy as np
import matplotlib.pyplot as plt

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


def read_correctness(csv_file):
    """
    Reads correctness.csv and returns lists of (n, c) for each task.
    """
    data = []
    if not os.path.exists(csv_file):
        print(f"Warning: File {csv_file} not found.")
        return data

    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            if not row: continue
            samples = [int(x) for x in row[1:] if x.strip()]
            n = len(samples)
            c = sum(samples)
            data.append((n, c))
            
    return data


def plot_pass_k(data, output_image, title, model_name):
    """
    Calculates and plots Pass@k.
    """
    if not data:
        return
        
    # Find max n
    max_n = max(d[0] for d in data)
    k_values = list(range(1, max_n + 1))
    pass_k_scores = []
    
    # Calculate global Pass@k
    for k in k_values:
        total_score = 0
        valid_tasks = 0
        for n, c in data:
            if n >= k:
                total_score += estimate_pass_at_k(n, c, k)
                valid_tasks += 1
        
        if valid_tasks > 0:
            pass_k_scores.append(total_score / valid_tasks)
        else:
            pass_k_scores.append(0.0)
            
    # 1. Plot Pass@k (Linear)
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, pass_k_scores, marker='o', linestyle='-', linewidth=2)
    plt.title(f"Pass@k ({model_name})")
    plt.xlabel("k (samples per problem)")
    plt.ylabel("Pass@k")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.ylim(0.0, 1.05)
    plt.tight_layout()
    plt.savefig(output_image, dpi=300)
    plt.close()
    print(f"Saved Pass@k plot to {output_image}")
    
    # 2. Plot 1 - Pass@k (Log-Log) with Power Law Fit
    failure_rates = [1.0 - s for s in pass_k_scores]
    min_k_fit = 10
    
    x_fit_candidates = []
    y_fit_candidates = []
    
    # First pass: collect all valid log points
    for k, fail in zip(k_values, failure_rates):
        if fail > 0:
            x_fit_candidates.append(k)
            y_fit_candidates.append(fail)
            
    # Apply high-k filter if we have enough points
    x_fit = [x for x in x_fit_candidates if x >= min_k_fit]
    y_fit = [y for x, y in zip(x_fit_candidates, y_fit_candidates) if x >= min_k_fit]
    
    if len(x_fit) < 3:
        x_fit = x_fit_candidates
        y_fit = y_fit_candidates
        fit_label_suffix = "(all k)"
    else:
        fit_label_suffix = f"(k>={min_k_fit})"
            
    if len(x_fit) > 1:
        log_x = np.log(x_fit)
        log_y = np.log(y_fit)
        
        # Fit linear regression
        slope, intercept = np.polyfit(log_x, log_y, 1)
        exponent = slope # This is 'b'
        
        # Generate fit line
        fit_y = np.exp(intercept) * (np.array(x_fit) ** slope)
        
        plt.figure(figsize=(10, 6))
        plt.loglog(x_fit_candidates, y_fit_candidates, marker='o', linestyle='', label='Data', alpha=0.5) 
        plt.loglog(x_fit, y_fit, marker='o', linestyle='', color='blue', label='Fit Data') 
        plt.loglog(x_fit, fit_y, linestyle='--', color='black', label=f'Fit: exponent = {exponent:.3f} {fit_label_suffix}')
        
        plt.title(f"Failure Rate (1 - Pass@k) Scaling ({model_name})")
        plt.xlabel("k (samples per problem)")
        plt.ylabel("1 - Pass@k")
        plt.grid(True, which="both", ls="-", alpha=0.2)
        plt.legend()
        plt.tight_layout()
        
        output_log = output_image.replace("pass_k.png", "pass_k_log_fit.png")
        plt.savefig(output_log, dpi=300)
        plt.close()
        print(f"Saved Log-Log fit plot to {output_log}")
    else:
        print("Not enough data points for log-log fit.")


def calculate_curve(tasks_list, metric="attempts"):
    """
    Calculates coverage curve for a given ORDERED list of tasks.
    Returns (x_values, y_values)
    """
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


def simulate_realizations(tasks_dict, n_realizations=100, metric="attempts"):
    """
    Generates N realizations by shuffling tasks and samples (Standard Random).
    """
    realization_curves = []
    task_keys = list(tasks_dict.keys())
    
    print(f"Simulating {n_realizations} Standard realizations for {metric}...")
    import random
    
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


def simulate_red_realizations(tasks_dict, n_realizations=100, metric="attempts"):
    """
    Generates N realizations for Reset and Discard (ReD) strategy.
    Strategy: Round-robin. In each round, shuffle remaining tasks, try next sample for each.
    """
    realization_curves = []
    task_keys = list(tasks_dict.keys())
    total_problems = len(task_keys)
    
    print(f"Simulating {n_realizations} ReD realizations for {metric}...")
    import random
    
    for _ in range(n_realizations):
        task_samples = {}
        for tid in task_keys:
            samps = tasks_dict[tid][:]
            # Only shuffle samples once per realization to define the 'deck'
            random.shuffle(samps)
            task_samples[tid] = samps
            
        unsolved_ids = task_keys[:]
        task_indices = {tid: 0 for tid in task_keys}
        
        cumulative_cost = 0
        solved_count = 0
        
        x = [0]
        y = [0.0]
        
        # Round Loop
        while unsolved_ids:
            random.shuffle(unsolved_ids)
            next_round_unsolved = []
            
            for tid in unsolved_ids:
                idx = task_indices[tid]
                samples = task_samples[tid]
                
                if idx >= len(samples):
                    continue
                
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


def read_full_data(correctness_file, tokens_file):
    """
    Reads both CSVs and returns a dictionary.
    """
    tasks = {}
    if not os.path.exists(correctness_file) or not os.path.exists(tokens_file):
        print("Warning: CSV files not found.")
        return {}
        
    print(f"Reading data from {correctness_file} and {tokens_file}...")
    
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
    
    print(f"Loaded {len(tasks)} tasks.")
    return tasks
