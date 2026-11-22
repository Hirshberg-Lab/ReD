import os
import argparse
import csv
from tqdm import tqdm
from human_eval.data import read_problems
from groq import Groq
from dotenv import load_dotenv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="results")
    args = parser.parse_args()
    
    load_dotenv()
    client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
    
    problems = read_problems()
    
    # Sanitize
    safe_model_name = args.model.replace("/", "_")
    output_file = os.path.join(args.output_dir, f"{safe_model_name}_prompt_tokens.csv")
    
    print(f"Measuring prompt tokens for {args.model} on {len(problems)} problems...")
    
    # Check if exists to avoid unnecessary work? No, user explicitly asked to run it.
    
    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["task_id", "prompt_tokens"])
        
        for task_id in tqdm(problems):
            problem = problems[task_id]
            prompt = problem["prompt"]
            
            messages = [
                {
                    "role": "system",
                    "content": "You are an intelligent coding assistant. You will be provided with a function signature and docstring. You must complete the function body. Return ONLY the python code for the function implementation. Do not add explanations or tests. Wrap the code in a python markdown block."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            try:
                # Max tokens 1 because we only care about the Input processing usage
                chat_completion = client.chat.completions.create(
                    messages=messages,
                    model=args.model,
                    temperature=0.0,
                    max_tokens=1,
                    top_p=1,
                    stop=None,
                    stream=False,
                )
                
                if chat_completion.usage:
                    p_tokens = chat_completion.usage.prompt_tokens
                    writer.writerow([task_id, p_tokens])
                else:
                    print(f"No usage returned for {task_id}")
                    writer.writerow([task_id, 0])
                    
            except Exception as e:
                print(f"Error processing {task_id}: {e}")
                writer.writerow([task_id, 0])
                
    print(f"Saved prompt tokens to {output_file}")

if __name__ == "__main__":
    main()
