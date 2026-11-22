"""Simulation of continuous reflection (CR) and of Reflection-ReD on the multi-turn reflection data. Used by
tables_s7_s8_reflection.py (Supplementary Tables 7 and 8).

Trimmed copy of simulate_reflection.py from the continuous_reflection branch of the ReD4LLMs repository: only the
functions that the paper's tables call are kept, with their code unchanged.
"""
import json
import random
from tqdm import tqdm

def load_reflection_matrix(filepath, in_price=None, out_price=None):
    """
    Custom loader for reflection data. 
    Groups by task_id, and then by chain_idx to preserve sequential state.
    """
    print(f"Loading stateful matrix from {filepath}...")
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
            cid = data.get("chain_idx", 0)
            
            if tid not in tasks:
                tasks[tid] = {}
            if cid not in tasks[tid]:
                tasks[tid][cid] = []
                
            tasks[tid][cid].append(data)
    
    # Ensure sequential order within each chain
    for tid in tasks:
        for cid in tasks[tid]:
            tasks[tid][cid].sort(key=lambda x: x.get("attempt_idx", 0))
            
    return tasks


def simulate_continuous_reflection(tasks, realizations, metric="usd_cost"):
    """
    Simulates multi-start sequential test-time scaling under a perfect verifier. 
    Evaluates a reflection chain turn-by-turn. If the chain fails, it starts a new independent chain,
    utilizing the full budget (e.g., 20 chains * 5 turns = 100 attempts) to fairly match ReD.
    """
    true_curves = []
    task_ids = list(tasks.keys())
    total_problems = len(task_ids)
    
    for _ in tqdm(range(realizations), desc=f"Reflection Sim ({metric})"):
        shuffled_tids = task_ids[:]
        random.shuffle(shuffled_tids)
        cost, solved = 0.0, 0
        x, y = [0.0], [0.0]
        
        for tid in shuffled_tids:
            # Shuffle the order of independent chains to simulate random restarts
            available_chains = list(tasks[tid].keys())
            random.shuffle(available_chains)
            
            problem_solved = False
            
            # Traverse multiple chains until success or the total budget is exhausted
            for cid in available_chains:
                chain = tasks[tid][cid]
                
                # Traverse the current chain sequentially
                for attempt in chain:
                    cost += attempt.get(metric, 1) 
                    
                    if attempt["passed"]:
                        solved += 1
                        x.append(cost)
                        y.append(solved / total_problems)
                        problem_solved = True
                        break # Break out of the attempt loop (success on this turn)
                        
                if problem_solved:
                    break # Break out of the chain loop (move to the next problem)
                    
            if not problem_solved:
                # All chains for this problem were exhausted without success
                x.append(cost)
                y.append(solved / total_problems)
                
        true_curves.append({"x": x, "y": y})
        
    return true_curves


def simulate_reflection_red(tasks, realizations, metric="usd_cost"):
    """
    Simulates Multi-Start Reflection combined with Breadth-First Reset-and-Discard (ReD).
    Evaluates exactly one attempt across all active problems at a time.
    If a stateful reflection chain fails its depth limit, it resets the context and starts a new independent chain from Turn 1.
    """
    red_curves = []
    task_ids = list(tasks.keys())
    total_problems = len(task_ids)
    
    for _ in tqdm(range(realizations), desc=f"Reflection-ReD Sim ({metric})"):
        cost, solved = 0.0, 0
        x, y = [0.0], [0.0]
        
        # Shuffle order of independent chains for each problem to simulate random restarts
        chains = {}
        for tid in task_ids:
            available_chains = list(tasks[tid].keys())
            random.shuffle(available_chains)
            chains[tid] = available_chains
            
        active_tids = list(task_ids)
        # Track pointers per problem {"c_idx": active chain, "t_idx": active turn in chain}
        pointers = {tid: {"c_idx": 0, "t_idx": 0} for tid in active_tids}
        
        while active_tids:
            # We shuffle the active_tids order in each round for theoretical rigour (standard ReD does this passively)
            shuffled_active = active_tids[:]
            random.shuffle(shuffled_active)
            
            for tid in shuffled_active:
                ptr = pointers[tid]
                cid = chains[tid][ptr["c_idx"]]
                chain = tasks[tid][cid]
                chain_len = len(chain)
                
                attempt = chain[ptr["t_idx"]]
                cost += attempt.get(metric, 1)
                
                if attempt["passed"]:
                    solved += 1
                    x.append(cost)
                    y.append(solved / total_problems)
                    active_tids.remove(tid)
                else:
                    ptr["t_idx"] += 1
                    if ptr["t_idx"] >= chain_len:
                        # Chain failed its depth limit! Execute Hard Context Reset
                        ptr["c_idx"] += 1
                        ptr["t_idx"] = 0
                        
                        if ptr["c_idx"] >= len(chains[tid]):
                            # Exhausted all multi-start reflection chains
                            active_tids.remove(tid)
                            x.append(cost)
                            y.append(solved / total_problems)
                            
        red_curves.append({"x": x, "y": y})
        
    return red_curves
