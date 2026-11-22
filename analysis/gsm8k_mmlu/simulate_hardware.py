"""GPU-time (KV-cache) model of solve-to-completion (prefill paid once per problem) and strict ReD (prefill paid on
every attempt). Used by table_s10_kv_cache.py (Supplementary Table 10).

Trimmed copy of simulate_hardware.py from the KV_cache branch of the ReD4LLMs repository: only the functions that the
paper's table calls are kept, with their code unchanged (the micro-batched variant is not in the paper).
"""
import json
import random
from tqdm import tqdm

def load_matrix(filepath, tolerate_uneven=False):
    print(f"Loading matrix from {filepath}...")
    tasks = {}
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            tid = data["task_id"]
            if tid not in tasks:
                tasks[tid] = []
            tasks[tid].append(data)
            
    if not tasks:
        return tasks, 0

    if tolerate_uneven:
        target_k = max(len(attempts) for attempts in tasks.values())
        for tid, attempts in tasks.items():
            current_len = len(attempts)
            if current_len < target_k:
                avg_in = sum(a.get("in_tokens", 0) for a in attempts) / current_len if current_len > 0 else 0
                avg_out = sum(a.get("out_tokens", 0) for a in attempts) / current_len if current_len > 0 else 0
                padding = [{
                    "task_id": tid,
                    "attempt_idx": i,
                    "passed": False,
                    "in_tokens": avg_in,
                    "out_tokens": avg_out,
                    "synthetic_padding": True
                } for i in range(current_len, target_k)]
                tasks[tid].extend(padding)
        return tasks, target_k
    else:
        target_k = min(len(attempts) for attempts in tasks.values())
        for tid in tasks:
            tasks[tid] = tasks[tid][:target_k]
        return tasks, target_k


def simulate_hardware_standard(tasks, realizations, prefill_ms, decode_ms, fpr=0.0, fnr=0.0):
    true_curves = []
    perceived_curves = []
    task_ids = list(tasks.keys())
    total_problems = len(task_ids)
    
    for _ in tqdm(range(realizations), desc="Hardware Sim: Standard"):
        shuffled_tids = task_ids[:]
        random.shuffle(shuffled_tids)
        gpu_seconds, solved, perceived_solved = 0.0, 0, 0
        x, y, y_perc = [0.0], [0.0], [0.0]
        
        for tid in shuffled_tids:
            attempts = tasks[tid][:]
            random.shuffle(attempts)
            
            # KV Cache tracks if the prefill tax has been paid for this problem
            cache_hit = False 
            
            for attempt in attempts:
                in_tokens = attempt.get("in_tokens", 0)
                out_tokens = attempt.get("out_tokens", 0)
                actual_pass = attempt["passed"]
                
                # Apply Hardware Latency Model
                if not cache_hit:
                    gpu_seconds += (in_tokens * prefill_ms) / 1000.0
                    cache_hit = True # Subsequent attempts inherit the cache
                gpu_seconds += (out_tokens * decode_ms) / 1000.0
                
                # Verifier Logic
                if actual_pass:
                    verifier_pass = random.random() >= fnr
                else:
                    verifier_pass = random.random() < fpr
                
                if verifier_pass:
                    perceived_solved += 1
                    if actual_pass:
                        solved += 1
                    x.append(gpu_seconds)
                    y.append(solved / total_problems)
                    y_perc.append(perceived_solved / total_problems)
                    break
            else:
                x.append(gpu_seconds)
                y.append(solved / total_problems)
                y_perc.append(perceived_solved / total_problems)
                
        true_curves.append({"x": x, "y": y})
        perceived_curves.append({"x": x, "y": y_perc})
        
    return true_curves, perceived_curves


def simulate_hardware_strict_red(tasks, realizations, prefill_ms, decode_ms, fpr=0.0, fnr=0.0):
    true_curves = []
    perceived_curves = []
    task_ids = list(tasks.keys())
    total_problems = len(task_ids)
    min_k = len(tasks[task_ids[0]])
    
    for _ in tqdm(range(realizations), desc="Hardware Sim: Strict ReD"):
        task_decks = {tid: random.sample(tasks[tid], len(tasks[tid])) for tid in task_ids}
        unsolved_ids = task_ids[:]
        gpu_seconds, solved, perceived_solved = 0.0, 0, 0
        x, y, y_perc = [0.0], [0.0], [0.0]
        
        for round_idx in range(min_k):
            if not unsolved_ids: break
            random.shuffle(unsolved_ids)
            next_unsolved = []
            for tid in unsolved_ids:
                attempt = task_decks[tid][round_idx]
                in_tokens = attempt.get("in_tokens", 0)
                out_tokens = attempt.get("out_tokens", 0)
                actual_pass = attempt["passed"]
                
                # Apply Hardware Latency Model (0% Cache Hit, constant context switching)
                gpu_seconds += (in_tokens * prefill_ms) / 1000.0
                gpu_seconds += (out_tokens * decode_ms) / 1000.0
                
                # Verifier Logic
                if actual_pass:
                    verifier_pass = random.random() >= fnr
                else:
                    verifier_pass = random.random() < fpr
                
                if verifier_pass:
                    perceived_solved += 1
                    if actual_pass:
                        solved += 1
                    x.append(gpu_seconds)
                    y.append(solved / total_problems)
                    y_perc.append(perceived_solved / total_problems)
                else:
                    next_unsolved.append(tid)
            unsolved_ids = next_unsolved
            
        if x[-1] != gpu_seconds:
            x.append(gpu_seconds)
            y.append(solved / total_problems)
            y_perc.append(perceived_solved / total_problems)
            
        true_curves.append({"x": x, "y": y})
        perceived_curves.append({"x": x, "y": y_perc})
        
    return true_curves, perceived_curves
