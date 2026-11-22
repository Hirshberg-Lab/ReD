import os
import re
import json
import argparse
import threading
import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datasets import load_dataset
from tqdm import tqdm
from dotenv import load_dotenv
from groq import Groq

# Load environment variables
load_dotenv()

# Thread lock for safe writing to the JSONL file
write_lock = threading.Lock()

def get_model_pricing(model_name: str):
    """
    Returns the pricing per 1M tokens (input, output) for Groq models.
    Defaulting to Llama 3 8B pricing as a fallback.
    """
    m = model_name.lower()
    if "70b" in m:
        return 0.59, 0.79
    elif "8b" in m:
        return 0.05, 0.08
    elif "mixtral" in m:
        return 0.24, 0.24
    else:
        return 0.05, 0.08  # Default

def load_dataset_by_name(dataset_name: str):
    """
    Dynamically loads and formats the requested dataset.
    """
    problems = {}
    
    if dataset_name == "gsm8k":
        print("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main", split="test")
        for i, item in enumerate(dataset):
            exact_answer = item['answer'].split('#### ')[1].strip().replace(',', '')
            problems[str(i)] = {
                "question": item['question'],
                "exact_answer": exact_answer
            }
            
    elif dataset_name.startswith("mmlu-pro"):
        print(f"Loading {dataset_name.upper()} dataset...")
        dataset = load_dataset("TIGER-Lab/MMLU-Pro", split="test")
        letters = "ABCDEFGHIJ"
        
        limit = int(dataset_name.split("-")[-1]) if dataset_name.split("-")[-1].isdigit() else float('inf')
        
        for i, item in enumerate(dataset):
            if i >= limit: break
            
            question = item['question']
            options = item['options']
            formatted_options = "\n".join([f"{letters[j]}) {opt}" for j, opt in enumerate(options)])
            full_prompt = f"{question}\n\nOptions:\n{formatted_options}"
            
            problems[str(i)] = {
                "question": full_prompt,
                "exact_answer": item['answer'] 
            }
            
    return problems

def extract_answer(text: str, dataset_name: str) -> str:
    """
    Strict regex extraction of the final answer from the LLM output based on dataset format.
    """
    if not text:
        return None
        
    if dataset_name == "gsm8k":
        match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')
            
    elif dataset_name.startswith("mmlu-pro"):
        match = re.search(r'####\s*([A-J])', text, re.IGNORECASE)
        if match:
            return match.group(1).upper()
            
    return None

def process_problem_chain(client, model, temperature, task_id, chain_idx, problem, max_turns, pricing, dataset_name):
    """
    Executes a single, continuous, stateful reflection sequence (a chain) for a problem.
    Stops early if the model achieves the correct answer.
    """
    in_price_per_m, out_price_per_m = pricing
    prompt = problem["question"]
    exact_answer = problem["exact_answer"]
    
    if dataset_name == "gsm8k":
        sys_content = "You are a mathematical reasoning assistant. Solve the problem step-by-step. You MUST conclude your response with the exact final numerical answer prefixed by '#### '. For example: '#### 42'."
    elif dataset_name.startswith("mmlu-pro"):
        sys_content = "You are an expert reasoning assistant. Solve the problem step-by-step. You MUST conclude your response with the exact final letter choice prefixed by '#### '. For example: '#### B'."
        
    messages = [
        {"role": "system", "content": sys_content},
        {"role": "user", "content": prompt}
    ]
    
    results = []
    max_retries = 5
    base_delay = 2.0
    
    for attempt_idx in range(max_turns):
        success = False
        
        for attempt_num in range(max_retries):
            try:
                resp = client.chat.completions.create(
                    messages=messages,
                    model=model,
                    temperature=temperature,
                    max_tokens=1024
                )
                
                output_text = resp.choices[0].message.content
                in_tokens = resp.usage.prompt_tokens if resp.usage else 0
                out_tokens = resp.usage.completion_tokens if resp.usage else 0
                usd_cost = (in_tokens / 1_000_000) * in_price_per_m + (out_tokens / 1_000_000) * out_price_per_m
                
                model_answer = extract_answer(output_text, dataset_name)
                passed = (model_answer == exact_answer)
                
                attempt_data = {
                    "task_id": task_id,
                    "chain_idx": chain_idx,
                    "attempt_idx": attempt_idx,
                    "passed": passed,
                    "in_tokens": in_tokens,
                    "out_tokens": out_tokens,
                    "usd_cost": usd_cost,
                    "error": None
                }
                
                # Write immediately to preserve data in case of a crash
                with write_lock:
                    with open(output_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(attempt_data) + "\n")
                        
                results.append(attempt_data)
                
                if passed:
                    # Early stop: The model got it right, no need to reflect further on this chain.
                    return results
                    
                else:
                    # Reflection routing: Append context for the next iteration
                    messages.append({"role": "assistant", "content": output_text})
                    reflection_prompt = f"Your extracted answer '{model_answer}' is incorrect. Please reflect on your previous reasoning, identify the mistake, and try again. You MUST conclude your response with the exact final answer prefixed by '#### '."
                    messages.append({"role": "user", "content": reflection_prompt})
                    
                success = True
                break # Break retry loop, proceed to next attempt_idx (turn)
                
            except Exception as e:
                if attempt_num < max_retries - 1:
                    sleep_time = base_delay * (2 ** attempt_num) + random.uniform(0, 1)
                    time.sleep(sleep_time)
                    continue
                else:
                    # Hard fail on API after retries
                    with write_lock:
                        print(f"\nTask {task_id} (Chain {chain_idx}) failed entirely on turn {attempt_idx}: {str(e)}")
                    return results
        
        if not success:
            break
            
    return results

def main():
    parser = argparse.ArgumentParser(description="Generate Stateful Reflection Matrix via Groq API")
    parser.add_argument("--model", type=str, required=True, help="Groq model ID")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "mmlu-pro", "mmlu-pro-500"], default="gsm8k", help="Target benchmark")
    parser.add_argument("--n_chains", type=int, default=10, help="Number of independent reflection chains per problem")
    parser.add_argument("--max_turns", type=int, default=5, help="Max multi-turn attempts (depth) per chain")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent API workers")
    args = parser.parse_args()

    if args.temperature <= 0.0:
        print("Error: Temperature must be > 0.0 to ensure variance across independent chains.")
        return

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment.")
        return

    client = Groq(api_key=api_key)
    pricing = get_model_pricing(args.model)
    
    target_dir = f"results_reflection_{args.dataset}"
    os.makedirs(target_dir, exist_ok=True)
    
    safe_model_name = args.model.replace("/", "_")
    
    # Needs to be global for the thread lock writer
    global output_file
    output_file = os.path.join(target_dir, f"{safe_model_name}_reflection_matrix.jsonl")

    problems = load_dataset_by_name(args.dataset)
    
    # Resumption Logic tracking (task_id, chain_idx)
    completed_tasks = set()
    if os.path.exists(output_file):
        print(f"Found existing reflection matrix at {output_file}. Scanning for completed chains...")
        task_counts = {}
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                tid = data["task_id"]
                chain_idx = data.get("chain_idx", 0)
                key = (tid, chain_idx)
                
                if data.get("passed"):
                    completed_tasks.add(key)
                else:
                    task_counts[key] = task_counts.get(key, 0) + 1
                    if task_counts[key] >= args.max_turns:
                        completed_tasks.add(key)
                        
    pending_tasks = []
    for tid in problems.keys():
        for chain_idx in range(args.n_chains):
            if (tid, chain_idx) not in completed_tasks:
                pending_tasks.append((tid, chain_idx))
    
    print(f"Total problem-chains required: {len(problems) * args.n_chains}")
    print(f"Chains already completed: {len(completed_tasks)}")
    print(f"Chains pending generation: {len(pending_tasks)}")
    
    if not pending_tasks:
        print("Reflection matrix generation is already complete!")
        return

    random.shuffle(pending_tasks)
    print(f"Starting generation with {args.workers} concurrent workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_problem_chain, client, args.model, args.temperature, tid, chain_idx, problems[tid], args.max_turns, pricing, args.dataset
            ): (tid, chain_idx) for (tid, chain_idx) in pending_tasks
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {args.dataset.upper()} Reflection"):
            _ = future.result() 

    print(f"Reflection matrix generation complete. Data appended to {output_file}")

if __name__ == "__main__":
    main()