"""Library functions used by plot_figures.py (Fig. 5c, cost in USD).

Trimmed copy of plotting/plot_paper_figure_comparison.py from the ReD repository: only the functions that the
paper's figures and tables call are kept, with their code unchanged.
"""
import os
import csv
import random

def estimate_prompt_cost(problems_dict, model_name, output_dir="results"):
    m = model_name.lower()
    if "8b" in m: price_per_m = 0.05
    elif "20b" in m or "gpt-oss" in m: price_per_m = 0.075
    else: price_per_m = 0.59
    
    costs = {}
    safe_name = model_name.replace("/", "_")
    prompt_file = os.path.join(output_dir, f"{safe_name}_prompt_tokens.csv")
    if os.path.exists(prompt_file):
        with open(prompt_file, 'r') as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                if not row: continue
                tid = row[0]
                ptoks = int(row[1])
                costs[tid] = (ptoks / 1_000_000) * price_per_m
    else:
        # Heuristic fallback if file missing
        for tid, prob in problems_dict.items():
            if tid not in costs:
                prompt = prob["prompt"]
                tokens = (len(prompt) / 3.5) * 2.0
                costs[tid] = (tokens / 1_000_000) * price_per_m
    return costs


def get_completion_price_per_m(model_name):
    m = model_name.lower()
    if "8b" in m: return 0.08
    elif "20b" in m or "gpt-oss" in m: return 0.30
    else: return 0.79


def calculate_usd_curve(tasks_list_tuples, prompt_costs, completion_price_per_m):
    total_problems = len(tasks_list_tuples)
    cumulative_cost = 0.0
    solved_count = 0
    x = [0.0]
    y = [0.0]
    
    for tid, samples in tasks_list_tuples:
        solved_this_task = False
        p_cost = prompt_costs.get(tid, 0.0)
        
        for s in samples:
            c_tokens = s.get('tokens', 0)
            c_cost = (c_tokens / 1_000_000) * completion_price_per_m
            step_cost = p_cost + c_cost
            cumulative_cost += step_cost
            
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


def simulate_realizations_usd(tasks_dict, n_realizations, prompt_costs, comp_price):
    realization_curves = []
    task_keys = list(tasks_dict.keys())
    
    for _ in range(n_realizations):
        shuffled_keys = task_keys[:]
        random.shuffle(shuffled_keys)
        
        current_task_list = []
        for tid in shuffled_keys:
            samples = tasks_dict[tid][:]
            random.shuffle(samples)
            current_task_list.append((tid, samples))
            
        rx, ry = calculate_usd_curve(current_task_list, prompt_costs, comp_price)
        realization_curves.append((rx, ry))
    return realization_curves


def simulate_red_realizations_usd(tasks_dict, n_realizations, prompt_costs, comp_price):
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
        
        cumulative_cost = 0.0
        solved_count = 0
        x = [0.0]
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
                
                # Cost Calculation
                p_cost = prompt_costs.get(tid, 0.0)
                c_tokens = s.get('tokens', 0)
                c_cost = (c_tokens / 1_000_000) * comp_price
                
                step_cost = p_cost + c_cost
                cumulative_cost += step_cost
                
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
