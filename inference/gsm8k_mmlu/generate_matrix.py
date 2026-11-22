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

# Sentinel for logged predictions where extraction found no well-formed option.
# Never a valid MMLU-Pro letter or GSM8K number, so it can't be mistaken for a
# real answer downstream (e.g. in majority-vote / self-consistency analysis).
UNK_SENTINEL = "UNK"

def get_model_pricing(model_name: str):
    """
    Returns the pricing per 1M tokens (input, output) for Groq models.
    Defaulting to Llama 3 8B pricing as a fallback.
    """
    m = model_name.lower()
    if "gpt-oss-120b" in m:
        return 0.15, 0.75
    elif "gpt-oss-20b" in m:
        return 0.10, 0.50
    elif "70b" in m:
        return 0.59, 0.79
    elif "8b" in m:
        return 0.05, 0.08
    elif "mixtral" in m:
        return 0.24, 0.24
    else:
        return 0.05, 0.08  # Default

def is_reasoning_model(model_name: str) -> bool:
    """
    Groq reasoning models (e.g. gpt-oss) emit a separate reasoning trace
    alongside the final answer, and accept reasoning_effort/reasoning_format.
    """
    return "gpt-oss" in model_name.lower()

def load_dataset_by_name(dataset_name: str):
    """
    Dynamically loads and formats the requested dataset.
    """
    problems = {}
    
    if dataset_name == "gsm8k":
        print("Loading GSM8K dataset...")
        dataset = load_dataset("gsm8k", "main", split="test")
        for i, item in enumerate(dataset):
            # GSM8K ground truth format: [reasoning] #### [answer]
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
            # Format options cleanly for the prompt
            formatted_options = "\n".join([f"{letters[j]}) {opt}" for j, opt in enumerate(options)])
            full_prompt = f"{question}\n\nOptions:\n{formatted_options}"
            
            problems[str(i)] = {
                "question": full_prompt,
                "exact_answer": item['answer'] # Ground truth is just 'A', 'B', etc.
            }
            
    return problems

def extract_answer(text: str, dataset_name: str) -> str:
    """
    Strict regex extraction of the final answer from the LLM output based on dataset format.
    """
    if dataset_name == "gsm8k":
        # Look for #### followed by optional spaces, then a number (handling commas and decimals)
        match = re.search(r'####\s*(-?\d+(?:,\d+)*(?:\.\d+)?)', text)
        if match:
            return match.group(1).replace(',', '')
            
    elif dataset_name.startswith("mmlu-pro"):
        # Look for #### followed by optional spaces, then a single letter A-J.
        # Take the LAST match, not the first: reasoning models can restate the
        # format mid-explanation before their true final answer.
        matches = re.findall(r'####\s*([A-J])', text, re.IGNORECASE)
        if matches:
            return matches[-1].upper()
            
    return None

def process_attempt(client, model, temperature, task_id, problem, attempt_idx, pricing, dataset_name, reasoning_effort="medium"):
    """
    Makes the API call, evaluates the output, and calculates the cost for a single attempt.
    """
    in_price_per_m, out_price_per_m = pricing

    prompt = problem["question"]
    exact_answer = problem["exact_answer"]

    # Dynamic System Prompt Routing
    if dataset_name == "gsm8k":
        sys_content = "You are a mathematical reasoning assistant. Solve the problem step-by-step. You MUST conclude your response with the exact final numerical answer prefixed by '#### '. For example: '#### 42'."
    elif dataset_name.startswith("mmlu-pro"):
        sys_content = "You are an expert reasoning assistant. Solve the problem step-by-step. You MUST conclude your response with the exact final letter choice prefixed by '#### '. For example: '#### B'."

    messages = [
        {
            "role": "system",
            "content": sys_content
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    reasoning = is_reasoning_model(model)
    extra_kwargs = {}
    if reasoning:
        # reasoning_format="parsed" splits the CoT into message.reasoning and
        # leaves message.content holding ONLY the final answer, so extraction
        # never sees the reasoning trace. Budget is raised because on gpt-oss
        # the reasoning tokens are drawn from the same completion budget as
        # the final answer.
        extra_kwargs["reasoning_effort"] = reasoning_effort
        extra_kwargs["reasoning_format"] = "parsed"

    max_retries = 5
    base_delay = 2.0

    for attempt_num in range(max_retries):
        try:
            # Note: Temperature must be > 0.0 to generate statistical variance for Pass@k
            resp = client.chat.completions.create(
                messages=messages,
                model=model,
                temperature=temperature,
                max_tokens=8192 if reasoning else 1024,
                **extra_kwargs
            )

            output_text = resp.choices[0].message.content or ""
            finish_reason = resp.choices[0].finish_reason
            in_tokens = resp.usage.prompt_tokens if resp.usage else 0
            out_tokens = resp.usage.completion_tokens if resp.usage else 0

            # Calculate Cost
            usd_cost = (in_tokens / 1_000_000) * in_price_per_m + (out_tokens / 1_000_000) * out_price_per_m

            # Evaluate — scoring is untouched: passed is computed from the raw
            # extractor output exactly as before. "predicted" below is a separate
            # logging-only field (None -> UNK_SENTINEL) for downstream analysis
            # (majority-vote / self-consistency / verifier-noise); it never feeds
            # back into passed.
            model_answer = extract_answer(output_text, dataset_name)
            passed = (model_answer == exact_answer)
            predicted_logged = model_answer if model_answer is not None else UNK_SENTINEL

            return {
                "task_id": task_id,
                "attempt_idx": attempt_idx,
                "passed": passed,
                "in_tokens": in_tokens,
                "out_tokens": out_tokens,
                "usd_cost": usd_cost,
                "finish_reason": finish_reason,
                "predicted": predicted_logged,
                "gold": exact_answer,
                "error": None
            }

        except Exception as e:
            if attempt_num < max_retries - 1:
                # Exponential backoff with jitter
                sleep_time = base_delay * (2 ** attempt_num) + random.uniform(0, 1)
                time.sleep(sleep_time)
                continue
            else:
                return {
                    "task_id": task_id,
                    "attempt_idx": attempt_idx,
                    "passed": False,
                    "in_tokens": 0,
                    "out_tokens": 0,
                    "usd_cost": 0.0,
                    "error": str(e)
                }

def main():
    parser = argparse.ArgumentParser(description="Generate Results Matrix via Groq API")
    parser.add_argument("--model", type=str, required=True, help="Groq model ID (e.g., llama-3.3-70b-versatile)")
    parser.add_argument("--dataset", type=str, choices=["gsm8k", "mmlu-pro", "mmlu-pro-500"], default="gsm8k", help="Target benchmark")
    parser.add_argument("--k_samples", type=int, default=100, help="Number of attempts per problem")
    parser.add_argument("--temperature", type=float, default=0.7, help="Generation temperature (Must be > 0)")
    parser.add_argument("--workers", type=int, default=10, help="Number of concurrent API requests")
    parser.add_argument("--reasoning_effort", type=str, choices=["low", "medium", "high"], default="medium", help="Reasoning effort for reasoning-capable models (e.g. gpt-oss). Ignored by non-reasoning models.")
    args = parser.parse_args()

    if args.temperature <= 0.0:
        print("Error: Temperature must be > 0.0 to generate variance for Pass@k simulation.")
        return

    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("Error: GROQ_API_KEY not found in environment.")
        return

    client = Groq(api_key=api_key)
    pricing = get_model_pricing(args.model)
    
    # Dynamic Folder Generation
    target_dir = f"results_{args.dataset}"
    os.makedirs(target_dir, exist_ok=True)
    
    safe_model_name = args.model.replace("/", "_")
    if is_reasoning_model(args.model):
        # Namespace by effort so low/medium runs never share (and silently
        # cross-contaminate via resumption) the same matrix file.
        safe_model_name = f"{safe_model_name}_{args.reasoning_effort}"
    output_file = os.path.join(target_dir, f"{safe_model_name}_matrix.jsonl")

    problems = load_dataset_by_name(args.dataset)
    
    # --- Resumption Logic ---
    # Track which (task_id, attempt_idx) have already been successfully generated
    completed_tasks = set()
    if os.path.exists(output_file):
        print(f"Found existing matrix at {output_file}. Scanning for completed attempts...")
        with open(output_file, 'r') as f:
            for line in f:
                data = json.loads(line)
                if data.get("error") is None: # Only count successful API calls
                    completed_tasks.add((data["task_id"], data["attempt_idx"]))
    
    # Build queue of pending tasks
    pending_tasks = []
    for task_id, problem in problems.items():
        for i in range(args.k_samples):
            if (task_id, i) not in completed_tasks:
                pending_tasks.append((task_id, problem, i))
                
    print(f"Total tasks required: {len(problems) * args.k_samples}")
    print(f"Tasks already completed: {len(completed_tasks)}")
    print(f"Tasks pending: {len(pending_tasks)}")
    
    if not pending_tasks:
        print("Matrix generation is already complete!")
        return

    # --- Shuffle the tasks to avoid API throttling ---
    random.shuffle(pending_tasks)

    # --- Concurrent Generation ---
    print(f"Starting generation with {args.workers} concurrent workers...")
    
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {
            executor.submit(
                process_attempt, client, args.model, args.temperature, task_id, problem, attempt_idx, pricing, args.dataset, args.reasoning_effort
            ): (task_id, attempt_idx)
            for (task_id, problem, attempt_idx) in pending_tasks
        }
        
        # Process as they complete
        with open(output_file, 'a', encoding='utf-8') as f:
            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Generating {args.dataset.upper()} Matrix"):
                result = future.result()
                
                # Write to disk immediately (thread-safe)
                if result.get("error") is None:
                    with write_lock:
                        f.write(json.dumps(result) + "\n")
                        f.flush()
                else:
                    print(f"\nTask {result['task_id']} attempt {result['attempt_idx']} failed entirely after retries: {result['error']}")

    print(f"Matrix generation complete. Data appended to {output_file}")

if __name__ == "__main__":
    main()