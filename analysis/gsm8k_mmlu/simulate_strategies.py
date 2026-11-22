"""Simulation of solve-to-completion (STC, "standard") and ReD on an N x k attempt matrix, with a perfect or
imperfect verifier. Used by tables_gsm8k_mmlu.py (Supplementary Tables 1, 2, 4, 5).

Trimmed copy of simulate_strategies.py from the ReD4LLMs repository: only the functions that the paper's tables
call are kept, with their code unchanged.
"""
import json
import random
from tqdm import tqdm

def load_matrix(filepath, tolerate_uneven=False, in_price=None, out_price=None):
    print(f"Loading matrix from {filepath}...")
    tasks = {}
    
    recompute_cost = (in_price is not None and out_price is not None)
    if recompute_cost:
        print(f"Retroactive pricing ON: Input ${in_price}/1M | Output ${out_price}/1M")

    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            
            if recompute_cost:
                data["usd_cost"] = (data.get("in_tokens", 0) / 1_000_000) * in_price + \
                                   (data.get("out_tokens", 0) / 1_000_000) * out_price
                
            tid = data["task_id"]
            if tid not in tasks:
                tasks[tid] = []
            tasks[tid].append(data)
    
    if not tasks:
        return tasks, 0

    if tolerate_uneven:
        target_k = max(len(attempts) for attempts in tasks.values())
        padded_count = 0
        for tid, attempts in tasks.items():
            current_len = len(attempts)
            if current_len < target_k:
                avg_in = sum(a.get("in_tokens", 0) for a in attempts) / current_len if current_len > 0 else 0
                avg_out = sum(a.get("out_tokens", 0) for a in attempts) / current_len if current_len > 0 else 0
                
                if recompute_cost:
                    avg_usd = (avg_in / 1_000_000) * in_price + (avg_out / 1_000_000) * out_price
                else:
                    avg_usd = sum(a.get("usd_cost", 0.0) for a in attempts) / current_len if current_len > 0 else 0.0
                
                padding = [{
                    "task_id": tid,
                    "attempt_idx": i,
                    "passed": False,
                    "in_tokens": avg_in,
                    "out_tokens": avg_out,
                    "usd_cost": avg_usd,
                    "synthetic_padding": True
                } for i in range(current_len, target_k)]
                
                tasks[tid].extend(padding)
                padded_count += len(padding)
        print(f"Tolerate Uneven ON: Padded {padded_count} missing attempts to reach k={target_k}.")
        return tasks, target_k
    else:
        target_k = min(len(attempts) for attempts in tasks.values())
        for tid in tasks:
            tasks[tid] = tasks[tid][:target_k]
        print(f"Strict Mode ON: Truncated all tasks to uniform k={target_k}.")
        return tasks, target_k


def simulate_standard(tasks, realizations, metric="usd_cost", fpr=0.0, fnr=0.0):
    true_curves = []
    perceived_curves = []
    task_ids = list(tasks.keys())
    total_problems = len(task_ids)
    
    for _ in tqdm(range(realizations), desc=f"Standard Sim ({metric})"):
        shuffled_tids = task_ids[:]
        random.shuffle(shuffled_tids)
        cost, solved, perceived_solved = 0.0, 0, 0
        x, y, y_perc = [0.0], [0.0], [0.0]
        
        for tid in shuffled_tids:
            attempts = tasks[tid][:]
            random.shuffle(attempts)
            for attempt in attempts:
                cost += attempt.get(metric, 1) 
                actual_pass = attempt["passed"]
                
                if actual_pass:
                    verifier_pass = random.random() >= fnr
                else:
                    verifier_pass = random.random() < fpr
                
                if verifier_pass:
                    perceived_solved += 1
                    if actual_pass:
                        solved += 1
                    x.append(cost)
                    y.append(solved / total_problems)
                    y_perc.append(perceived_solved / total_problems)
                    break
            else:
                x.append(cost)
                y.append(solved / total_problems)
                y_perc.append(perceived_solved / total_problems)
                
        true_curves.append({"x": x, "y": y})
        perceived_curves.append({"x": x, "y": y_perc})
        
    return true_curves, perceived_curves


def simulate_red(tasks, realizations, metric="usd_cost", fpr=0.0, fnr=0.0):
    true_curves = []
    perceived_curves = []
    task_ids = list(tasks.keys())
    total_problems = len(task_ids)
    min_k = len(tasks[task_ids[0]])
    
    for _ in tqdm(range(realizations), desc=f"ReD Sim ({metric})"):
        task_decks = {tid: random.sample(tasks[tid], len(tasks[tid])) for tid in task_ids}
        unsolved_ids = task_ids[:]
        cost, solved, perceived_solved = 0.0, 0, 0
        x, y, y_perc = [0.0], [0.0], [0.0]
        
        for round_idx in range(min_k):
            if not unsolved_ids: break
            random.shuffle(unsolved_ids)
            next_unsolved = []
            for tid in unsolved_ids:
                attempt = task_decks[tid][round_idx]
                cost += attempt.get(metric, 1)
                actual_pass = attempt["passed"]
                
                if actual_pass:
                    verifier_pass = random.random() >= fnr
                else:
                    verifier_pass = random.random() < fpr
                
                if verifier_pass:
                    perceived_solved += 1
                    if actual_pass:
                        solved += 1
                    x.append(cost)
                    y.append(solved / total_problems)
                    y_perc.append(perceived_solved / total_problems)
                else:
                    next_unsolved.append(tid)
            unsolved_ids = next_unsolved
            
        if x[-1] != cost:
            x.append(cost)
            y.append(solved / total_problems)
            y_perc.append(perceived_solved / total_problems)
            
        true_curves.append({"x": x, "y": y})
        perceived_curves.append({"x": x, "y": y_perc})
        
    return true_curves, perceived_curves
