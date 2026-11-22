import os
import re
import json
import csv
import time
import argparse
import threading
import concurrent.futures
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq
from human_eval.data import read_problems, write_jsonl
from human_eval.evaluation import evaluate_functional_correctness

# Load environment variables
load_dotenv()

def extract_code_block(text):
    """
    Extracts code from a markdown code block. 
    If no block is found, returns the text as is (fallback).
    Specific for Python code blocks used in HumanEval.
    """
    pattern = r"```python\n(.*?)```"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    
    # Try finding any code block if python specific one not found
    pattern_generic = r"```\n(.*?)```"
    match_generic = re.search(pattern_generic, text, re.DOTALL)
    if match_generic:
        return match_generic.group(1)
        
    return text

def main():
    parser = argparse.ArgumentParser(description="Reproduce HumanEval with Groq API")
    parser.add_argument("--model", type=str, required=True, help="Groq model ID (e.g., llama-3.1-8b-instant)")
    parser.add_argument("--details", action="store_true", help="Print generation details")
    parser.add_argument("--n_samples", type=int, default=1, help="Number of samples per problem")
    parser.add_argument("--k", type=str, default="1", help="Comma-separated k values for pass@k (e.g. '1,10')")
    parser.add_argument("--temperature", type=float, default=None, help="Temperature for generation")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of problems for testing")
    parser.add_argument("--no_save_samples", action="store_true", help="Do not save the full JSONL samples file (saves space)")
    parser.add_argument("--output_dir", type=str, default="results", help="Directory to save results")
    parser.add_argument("--reasoning_effort", type=str, default="medium", choices=["low", "medium", "high"],
                         help="Groq reasoning_effort. Only sent for gpt-oss models (ignored/omitted otherwise). "
                              "Default 'medium' matches the prior paper runs.")
    parser.add_argument("--workers", type=int, default=8, help="Parallel worker threads for generation calls")
    parser.add_argument("--max_retries", type=int, default=6, help="Max retries per generation call on transient/rate-limit errors")
    parser.add_argument("--skip_autoplot", action="store_true", help="Skip the automatic plot_coverage.py subprocess call at the end")
    args = parser.parse_args()

    # Parse k values
    try:
        k_values = [int(x) for x in args.k.split(",")]
    except ValueError:
        print("Error: --k must be comma-separated integers")
        return

    # Handle Temperature
    # If n_samples > 1 and temp is None, default to 0.8 (standard for pass@k)
    # If n_samples == 1 and temp is None, default to 0.0 (greedy)
    if args.temperature is None:
        if args.n_samples > 1:
            args.temperature = 0.8
            print(f"n_samples > 1, setting temperature to {args.temperature}")
        else:
            args.temperature = 0.0
            print(f"n_samples == 1, setting temperature to {args.temperature}")

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in .env or environment variables.")
        return

    client = Groq(api_key=api_key)

    # Sanitize model name for filenames
    safe_model_name = args.model.replace("/", "_")
    # Effort-suffixed tag for every output file, so low/medium/high sweep runs
    # never collide with each other -- or with the paper's existing un-suffixed
    # medium files, which this script must never overwrite.
    run_tag = f"{safe_model_name}_{args.reasoning_effort}"
    is_reasoning_model = "gpt-oss" in args.model.lower()

    problems = read_problems()

    if args.limit:
        problems = dict(list(problems.items())[:args.limit])

    # Ensure output dir exists (needed before the checkpoint file below)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    SYSTEM_PROMPT = ("You are an intelligent coding assistant. You will be provided with a "
                      "function signature and docstring. You must complete the function body. "
                      "Return ONLY the python code for the function implementation. Do not add "
                      "explanations or tests. Wrap the code in a python markdown block.")

    # Resumable checkpoint: every (task_id, sample_index) generation is appended
    # here as soon as it completes, so a crash or rate-limit kill mid-run loses
    # at most the in-flight requests, not everything already generated.
    raw_path = os.path.join(args.output_dir, f"{run_tag}_raw.jsonl")
    done = {}
    if os.path.exists(raw_path):
        with open(raw_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    done[(rec["task_id"], rec["sample_index"])] = rec
                except Exception:
                    pass
        print(f"Resuming from {raw_path}: {len(done)} (problem, attempt) cells already completed.")

    todo = [(tid, i) for tid in problems for i in range(args.n_samples) if (tid, i) not in done]
    print(f"Generating {len(todo)} samples ({len(done)} already done, target {args.n_samples}/problem) "
          f"for {len(problems)} problems using {args.model} [reasoning_effort={args.reasoning_effort}]...")

    write_lock = threading.Lock()

    def generate_one(task_id, sample_index):
        problem = problems[task_id]
        prompt = problem["prompt"]
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
        extra = {"reasoning_effort": args.reasoning_effort} if is_reasoning_model else {}

        delay = 2.0
        last_err = None
        for attempt in range(args.max_retries):
            try:
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=args.model,
                    temperature=args.temperature,
                    max_tokens=1024,
                    top_p=1,
                    stop=None,
                    stream=False,
                    **extra,
                )

                completion_text = chat_completion.choices[0].message.content

                # capture token usage
                tokens = 0
                p_tokens = 0
                if chat_completion.usage:
                    tokens = chat_completion.usage.completion_tokens
                    p_tokens = chat_completion.usage.prompt_tokens

                clean_code = extract_code_block(completion_text)

                if clean_code.strip().startswith(prompt.strip()):
                    clean_code = clean_code.replace(prompt.strip(), "", 1)

                rec = dict(task_id=task_id, completion=clean_code, tokens=tokens,
                           prompt_tokens=p_tokens, sample_index=sample_index)
                with write_lock:
                    with open(raw_path, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                return rec
            except Exception as e:
                last_err = e
                if attempt == args.max_retries - 1:
                    break
                time.sleep(delay)
                delay = min(delay * 2, 60)
        print(f"Giving up on {task_id} sample {sample_index} after {args.max_retries} attempts: {last_err}")
        return None

    if todo:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [executor.submit(generate_one, tid, i) for tid, i in todo]
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                future.result()

    # Reload the full set from the checkpoint so resumed and freshly-generated
    # cells are handled identically, then reconcile against the target grid.
    all_records = {}
    with open(raw_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            all_records[(rec["task_id"], rec["sample_index"])] = rec

    samples = []
    missing_cells = []
    for task_id in problems:
        for i in range(args.n_samples):
            key = (task_id, i)
            if key in all_records:
                samples.append(all_records[key])
            else:
                missing_cells.append(key)
                # Missing cell: fed into scoring as an empty completion, which
                # check_correctness fails naturally -- counted as a failure,
                # never silently dropped from the denominator.
                samples.append(dict(task_id=task_id, completion="", tokens=0, prompt_tokens=0, sample_index=i))

    if missing_cells:
        print(f"WARNING: {len(missing_cells)} (problem, attempt) cells missing after {args.max_retries} "
              f"retries each -- counted as FAILURES (0 tokens).")

    output_file = os.path.join(args.output_dir, f"{run_tag}_samples.jsonl")
    
    if not args.no_save_samples:
        write_jsonl(output_file, samples)
        print(f"Saved generations to {output_file}")
    else:
        print("Skipped saving full JSONL samples (--no_save_samples active).")
        
    # Save Prompt Tokens Metadata (One per task)
    # Since prompt tokens are constant per task, we just extract from the first sample of each task
    prompt_tokens_map = {}
    for s in samples:
        tid = s["task_id"]
        # Prefer the first real (non-missing) value so a missing sample_0 cell
        # doesn't zero out an otherwise-constant per-problem prompt token count.
        if tid not in prompt_tokens_map or (prompt_tokens_map[tid] == 0 and s.get("prompt_tokens", 0) > 0):
            prompt_tokens_map[tid] = s.get("prompt_tokens", 0)

    prompt_file = os.path.join(args.output_dir, f"{run_tag}_prompt_tokens.csv")
    with open(prompt_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "prompt_tokens"])
        for tid in sorted(prompt_tokens_map.keys()):
            writer.writerow([tid, prompt_tokens_map[tid]])
    print(f"Saved Prompt Token metadata to {prompt_file}")
    
    # Run evaluation and CSV generation
    print("Running evaluation and generating CSV reports...")
    
    # We need to run check_correctness on ALL samples to generate the CSVs
    # Pass@k can be calculated from these detailed results
    
    from human_eval.execution import check_correctness
    from human_eval.evaluation import estimate_pass_at_k

    # Function to check a single sample
    def check_sample(sample):
        prob = problems[sample["task_id"]]
        res = check_correctness(prob, sample["completion"], timeout=3.0)
        return {
            "task_id": sample["task_id"],
            "passed": res["passed"],
            "tokens": sample.get("tokens", 0),
            "sample_index": sample.get("sample_index", 0)
        }

    # Run checks in parallel
    print(f"Checking correctness for {len(samples)} samples...")
    detailed_results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
        futures = [executor.submit(check_sample, s) for s in samples]
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
            detailed_results.append(future.result())
            
    # Group results by task_id
    # task_id -> list of {passed, tokens, sample_index}
    grouped = {}
    for res in detailed_results:
        tid = res["task_id"]
        if tid not in grouped:
            grouped[tid] = []
        grouped[tid].append(res)
        
    # CRITICAL: Sort samples by sample_index to ensure "sample_0" is actually the first generated
    # This is required for correct Coverage@Cost analysis (accumulating costs sequentially)
    for tid in grouped:
        grouped[tid].sort(key=lambda x: x["sample_index"])
        
    # Sort samples consistently? 
    # We generated them in loop, but parallel execution scrambled order.
    # We should rely on the collected list order if possible, but we just need columns to represent "a sample".
    # Order doesn't strictly matter as samples are i.i.d., but good to be consistent.
    
    # Prepare CSV Data
    # Rows: task_id, sample_0, sample_1, ...
    
    # Find max samples to define columns
    max_samples = 0
    for tid in grouped:
        max_samples = max(max_samples, len(grouped[tid]))
        
    header = ["task_id"] + [f"sample_{i}" for i in range(max_samples)]
    
    csv_correctness_data = []
    csv_tokens_data = []
    
    # Sort task_ids for clean CSV
    sorted_tids = sorted(grouped.keys(), key=lambda x: int(x.split('/')[-1]))
    
    pass_counts = [] # for aggregate stats
    total_counts = []
    
    for tid in sorted_tids:
        # group is list of dicts
        group = grouped[tid]
        
        # passed row
        row_pass = [tid]
        # tokens row
        row_tok = [tid]
        
        c = 0
        for i in range(max_samples):
            if i < len(group):
                passed = 1 if group[i]["passed"] else 0
                tokens = group[i]["tokens"]
                row_pass.append(passed)
                row_tok.append(tokens)
                if passed: c += 1
            else:
                row_pass.append("")
                row_tok.append("")
        
        csv_correctness_data.append(row_pass)
        csv_tokens_data.append(row_tok)
        
        pass_counts.append(c)
        total_counts.append(len(group))

    # Write CSVs
    file_correctness = os.path.join(args.output_dir, f"{run_tag}_correctness.csv")
    with open(file_correctness, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_correctness_data)
        
    file_tokens = os.path.join(args.output_dir, f"{run_tag}_tokens.csv")
    with open(file_tokens, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(csv_tokens_data)
        
    print(f"Saved CSV reports to:\n - {file_correctness}\n - {file_tokens}")
    
    # Calculate and Print Pass@k
    # We have all the data now.
    
    print("Aggregate Results:")
    try:
        pass_at_k = {k: 0.0 for k in k_values}
        total_tasks = len(grouped)
        
        for i in range(total_tasks):
            n = total_counts[i]
            c = pass_counts[i]
            
            for k in k_values:
                if n >= k:
                    score = estimate_pass_at_k([n], [c], k)
                    pass_at_k[k] += float(score)
        
        print(f"Model: {args.model}")
        for k in k_values:
            pass_at_k[k] /= total_tasks
            print(f"Pass@{k}: {pass_at_k[k]:.2%}")
            
    except Exception as e:
        print(f"Error calculating Pass@k: {e}")

    # Calculate coverage analysis
    print("\nCalculating Coverage@Cost (Global Sequential)...")
    
    # User definition:
    # "Ask questions one by one and sum up the cost until each question was answered correctly for the first time."
    # This implies a sequential process: Problem 1 -> (try until success or exhaust) -> Problem 2 -> ...
    # Global Cost accumulates.
    
    coverage_rows = []
    
    # Sort task_ids to ensure deterministic order (e.g. HumanEval/0, HumanEval/1...)
    sorted_task_ids = sorted(grouped.keys(), key=lambda x: int(x.split('/')[-1]))
    
    total_problems = len(sorted_task_ids)
    
    # 1. Attempts
    # We assume "Attempt" cost is 1 per sample.
    cumulative_attempts = 0
    solved_count = 0
    
    rows_attempts = []
    # Initial point (0, 0)
    rows_attempts.append(["attempts", 0, 0.0])
    
    for tid in sorted_task_ids:
        samples_list = grouped[tid]
        # Iterate samples for this problem
        for s in samples_list:
            cumulative_attempts += 1
            if s["passed"]:
                solved_count += 1
                coverage = solved_count / total_problems
                rows_attempts.append(["attempts", cumulative_attempts, coverage])
                break # Move to next problem
        else:
            # If loop finished without break (not solved), we spent all attempts 
            # and move to next problem without increasing solved_count.
            # We record the cost point anyway to show flat lines?
            # User wants to know coverage at cost X.
            coverage = solved_count / total_problems
            rows_attempts.append(["attempts", cumulative_attempts, coverage])

    # 2. Tokens
    cumulative_tokens = 0
    solved_count_tok = 0
    
    rows_tokens = []
    rows_tokens.append(["tokens", 0, 0.0])
    
    for tid in sorted_task_ids:
        samples_list = grouped[tid]
        for s in samples_list:
            cumulative_tokens += s["tokens"]
            if s["passed"]:
                solved_count_tok += 1
                coverage = solved_count_tok / total_problems
                rows_tokens.append(["tokens", cumulative_tokens, coverage])
                break
        else:
            coverage = solved_count_tok / total_problems
            rows_tokens.append(["tokens", cumulative_tokens, coverage])
        
    # Save coverage CSVs
    
    # Save coverage CSVs
    
    # 1. Attempts
    file_cov_attempts = os.path.join(args.output_dir, f"{run_tag}_coverage_attempts.csv")
    with open(file_cov_attempts, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "budget", "coverage"])
        writer.writerows(rows_attempts)
        
    # 2. Tokens
    file_cov_tokens = os.path.join(args.output_dir, f"{run_tag}_coverage_tokens.csv")
    with open(file_cov_tokens, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "budget", "coverage"])
        writer.writerows(rows_tokens)
        
    print(f"Saved Coverage@Cost analysis to:\n - {file_cov_attempts}\n - {file_cov_tokens}")
    
    # Generate Plots using subprocess to ensure clean environment and independent execution
    # plot_coverage.py derives its own (un-suffixed) filenames from --model, so it can only
    # find this run's output when run_tag == safe_model_name (i.e. effort == medium AND the
    # caller didn't ask to skip it). For sweep runs, skip it -- analysis is done separately.
    if args.skip_autoplot or run_tag != safe_model_name:
        print(f"Skipping autoplot: files are named '{run_tag}_*', which plot_coverage.py "
              f"(--model {args.model}) does not know about. Run your own analysis script instead.")
    else:
        print("Generating comprehensive plots...")
        try:
            import subprocess
            import sys

            # Determine python executable
            python_exe = sys.executable

            # Run plot_coverage.py
            # We assume it is in the same directory
            script_path = os.path.join(os.path.dirname(__file__), "plot_coverage.py")

            cmd = [python_exe, script_path, "--model", args.model, "--output_dir", args.output_dir]

            print(f"Running plotting script: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            print("Plots generated successfully via plot_coverage.py.")

        except subprocess.CalledProcessError as e:
            print(f"Error running plotting script: {e}")
        except Exception as e:
            print(f"Error launching plotting script: {e}")

if __name__ == "__main__":
    main()
