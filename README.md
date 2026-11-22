# Resetting Improves the Fixed-Budget Inference of Large Language Models — code and data

This folder reproduces every data figure and table of the paper by S. Meir, T. D. Keidar, N. Levi, S. Reuveni and
B. Hirshberg. It contains the raw results of the LLM runs (which attempts were correct, and how many tokens each used)
and the scripts that turn them into the figures and tables. Reproducing the figures and tables needs no API key and no
network access.

**Reset-and-Discard (ReD)** spends a fixed budget of attempts across a pool of problems in rounds: in each round every
unsolved problem gets one attempt (τ = 1); a solved problem is discarded, an unsolved one is moved to the back of the
queue. **Solve-to-completion (STC)** attempts one problem at a time until it is solved (or its attempts run out) before
moving to the next. **coverage@cost** is the number of distinct problems solved for a given budget.

## Quick start

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
bash reproduce_all.sh          # about 8 minutes on 2 CPU cores
```

Figures are written to `outputs/figures/`, tables (Markdown) to `outputs/tables/`.

## What reproduces what

All scripts can be run from any directory; paths are resolved relative to this folder.

- **Figs. 4, 5, 6** (HumanEval): `analysis/humaneval/plot_figures.py` → `outputs/figures/fig{4,5,6}_*.pdf`.
- **Supplementary Fig. 1** (power-law fit of 1 − pass@k): `analysis/humaneval/suppfig1_pass_k_log_fit.py` →
  `outputs/figures/llama-3.1-8b-instant_pass_k_log_fit.png`.
- **Supplementary Tables:** the scripts that reproduce them are in `analysis/` (the files named `table*.py`). Each one
  writes its table to `outputs/tables/` and prints it.
- Figs. 1–3 are schematics drawn in the manuscript source; they contain no data.

`reproduce_all.sh` runs all of them.

**Randomness.** Every result is an average over random orders: in each realization the order of the problems and of
the attempts of each problem is shuffled, and the policies are simulated on that order. All scripts are seeded, so
their output is reproducible. The figures are identical to the paper. The original runs behind most tables were not
seeded, so a table's last digit can differ from the paper by about one standard error. Fig. 6
uses seed 311 (`--seed6`): the estimate of α varies between runs (over seeds 0–599, α = 0.33 ± 0.014, R² ≈ 0.95), and
seed 311 is one of the runs that gives the values in the paper.

## Data (`data/`)

| Folder | Benchmark | Models | Size | Format |
|---|---|---|---|---|
| `humaneval/` | HumanEval, 164 Python problems checked by unit tests | llama-3.1-8b-instant, llama-3.3-70b-versatile, openai/gpt-oss-20b (medium effort), gpt-oss-20b at low effort (`openai_gpt-oss-20b_low_*`) | 164 × 100 attempts | CSV |
| `gsm8k/` | GSM8K test set, 1,319 math problems | llama-3.1-8b-instant, llama-3.3-70b-versatile | 1,319 × 100 | JSONL |
| `gsm8k_reflection/` | GSM8K, continuous reflection | llama-3.1-8b-instant, llama-3.3-70b-versatile | 1,319 × 10 chains × up to 5 turns | JSONL |
| `mmlu-pro-500/` | MMLU-Pro, first 500 problems (multiple choice) | llama-3.1-8b-instant, llama-3.3-70b-versatile (500 × 100); gpt-oss-20b at low and medium effort (500 × 50) | | JSONL (+ NPZ) |
| `swebench/` | SWE-bench Lite, the 266 issues without flaky tests | DeepSeek-Coder-V2-Instruct in Moatless Tools (samples of Brown et al.) | 266 × up to 250 runs | CSV |

- HumanEval, `{model}_correctness.csv`: one row per problem, column `sample_j` = 1 if attempt j passed all unit tests.
  `{model}_tokens.csv`: output (completion) tokens of each attempt, including reasoning tokens for gpt-oss-20b.
  `{model}_prompt_tokens.csv`: input tokens of each problem's prompt (used for the USD cost in Fig. 5c).
  One attempt of gpt-oss-20b (medium) is missing and counted as a failure.
- GSM8K / MMLU-Pro, `{model}_matrix.jsonl`: one line per attempt with `task_id`, `attempt_idx`, `passed`,
  `in_tokens`, `out_tokens`, `usd_cost` (gpt-oss-20b also `finish_reason`, `predicted`, `gold`).
  The MMLU-Pro llama-3.1-8b file has 49,999 of 50,000 attempts; the simulation then uses the first 99 attempts of
  every problem. `openai_gpt-oss-20b_{low,medium}_matrix_matrices.npz` are the same gpt-oss-20b data as dense
  500 × 50 arrays (`correct`, `out_tokens`, `in_tokens`, `predicted`, `gold`, `seen`), built by
  `inference/gsm8k_mmlu/build_matrices.py`.
- GSM8K reflection, `{model}_reflection_matrix.jsonl`: one line per turn with `task_id`, `chain_idx`, `attempt_idx`
  (turn), `passed`, tokens and cost. A chain stops at its first correct answer. In the llama-3.1-8b file, 2,597 turns
  from an earlier generation run carry no `chain_idx`; the loader assigns them to chain 0, as in the runs behind the
  paper.
- SWE-bench Lite, `swebench_lite_deepseek_samples.csv`: one row per agent run with `issue`, `sample`, `resolved`,
  `n_steps` (agent steps), `n_calls` (model calls), `input_tokens`, `output_tokens`, `usd` (the cost the agent recorded,
  at $3 / $15 per million input / output tokens). Derived from the public samples of Brown et al. (2024),
  https://github.com/ScalingIntelligence/swe-bench-lite-samples, by `inference/swebench/extract_swebench_lite.py`.
  15 issues have 248 or 249 runs (the others crashed); the missing runs count as failed attempts with zero cost.

USD costs of the Groq runs use the Groq prices at the time of the runs, per million input/output tokens: llama-3.1-8b
$0.05/$0.08, llama-3.3-70b $0.59/$0.79, gpt-oss-20b $0.075/$0.30.

## How the raw data were generated (`inference/`)

The Groq scripts call the Groq API and need a key: copy `.env.example` to `.env` and set `GROQ_API_KEY`. They are not
needed to reproduce the figures and tables. Install with `pip install -r requirements-inference.txt`.

**HumanEval** (`inference/humaneval/run_standard_eval.py`): 100 independent attempts per problem at temperature 0.8,
top-p 1, max 1,024 output tokens. System prompt: *"You are an intelligent coding assistant. You will be provided with
a function signature and docstring. You must complete the function body. Return ONLY the python code for the function
implementation. Do not add explanations or tests. Wrap the code in a python markdown block."* The user message is the
HumanEval prompt (signature and docstring). The code is extracted from the markdown block (if it repeats the prompt,
the repeated part is removed) and checked with the official HumanEval unit tests.

```bash
python inference/humaneval/run_standard_eval.py --model llama-3.1-8b-instant --n_samples 100 --output_dir results
python inference/humaneval/run_standard_eval.py --model openai/gpt-oss-20b --n_samples 100 --reasoning_effort low --output_dir results
python inference/humaneval/measure_prompt_tokens.py --model llama-3.1-8b-instant --output_dir results
```

The script names its output files `{model}_{effort}_*` (for example `llama-3.1-8b-instant_medium_correctness.csv`);
the files in `data/humaneval/` from the main-text runs carry no effort suffix.

**GSM8K and MMLU-Pro** (`inference/gsm8k_mmlu/generate_matrix.py`), then `build_matrices.py` for the dense arrays.
The model is asked to solve the problem step by step and to end with the final answer after `#### ` (a number for
GSM8K, an option letter for MMLU-Pro); an attempt is correct if that answer matches the reference. Max 1,024 output
tokens and temperature 0.7 (the script's default) for the llama models; temperature 0.8 for gpt-oss-20b.

```bash
python inference/gsm8k_mmlu/generate_matrix.py --model llama-3.1-8b-instant --dataset gsm8k --k_samples 100 --temperature 0.7
python inference/gsm8k_mmlu/generate_matrix.py --model llama-3.1-8b-instant --dataset mmlu-pro-500 --k_samples 100 --temperature 0.7
python inference/gsm8k_mmlu/generate_matrix.py --model openai/gpt-oss-20b --dataset mmlu-pro-500 --k_samples 50 --temperature 0.8 --reasoning_effort low
python inference/gsm8k_mmlu/build_matrices.py --matrix results_mmlu-pro-500/openai_gpt-oss-20b_low_matrix.jsonl --k_samples 50
```

The gpt-oss-20b MMLU-Pro runs used a cap of 4,096 output tokens (7.4% of the medium-effort attempts reached it and
are scored as failures); the script now sets 8,192.

**Continuous reflection on GSM8K** (`inference/reflection/generate_reflection.py`): 10 independent chains per problem,
up to 5 turns per chain, temperature 0.7, max 1,024 output tokens. After a wrong
answer the model receives: *"Your extracted answer '…' is incorrect. Please reflect on your previous reasoning,
identify the mistake, and try again. You MUST conclude your response with the exact final answer prefixed by '#### '."*

```bash
python inference/reflection/generate_reflection.py --model llama-3.1-8b-instant --dataset gsm8k --n_chains 10 --max_turns 5
```

**SWE-bench Lite** (`inference/swebench/extract_swebench_lite.py`): no new runs; the script extracts the table in
`data/swebench/` from a clone of https://github.com/ScalingIntelligence/swe-bench-lite-samples (instructions in the
script).

## Folder structure

```
data/                    raw results (see above)
analysis/humaneval/      HumanEval: Figs. 4-6, Suppl. Fig. 1 and table scripts
analysis/gsm8k_mmlu/     GSM8K and MMLU-Pro table scripts
analysis/swebench/       SWE-bench Lite table script
                         (the other .py files in analysis/ hold the simulation functions the scripts use)
inference/               scripts that produced data/ (Groq API key needed, except swebench/)
reproduce_all.sh         runs every analysis script
```
