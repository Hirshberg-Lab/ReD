#!/usr/bin/env bash
# Reproduce every data figure and table of the paper from the raw results in data/.
# No API key or network access is needed. Outputs go to outputs/figures and outputs/tables.
# Run from anywhere:  bash reproduce_all.sh      (about 8 minutes on 2 CPU cores)
set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python3}

echo "== Figs. 4, 5, 6 (HumanEval) =="
$PY analysis/humaneval/plot_figures.py

echo "== Supplementary Fig. 1 (power-law fit of 1 - pass@k, llama-3.1-8b, HumanEval) =="
$PY analysis/humaneval/suppfig1_pass_k_log_fit.py

echo "== Supplementary Tables 1, 2, 4, 5 (GSM8K, MMLU-Pro; perfect and imperfect verifiers) =="
$PY analysis/gsm8k_mmlu/tables_gsm8k_mmlu.py

echo "== Supplementary Table 3 (SWE-bench Lite) =="
$PY analysis/swebench/table_s3_swebench.py

echo "== Supplementary Table 6 (HumanEval: imperfect verifiers) =="
$PY analysis/humaneval/table_s6_humaneval_imperfect.py

echo "== Supplementary Tables 7, 8 (GSM8K: continuous reflection, Reflection-ReD) =="
$PY analysis/gsm8k_mmlu/tables_s7_s8_reflection.py

echo "== Supplementary Table 10 (GSM8K: GPU time with KV cache) =="
$PY analysis/gsm8k_mmlu/table_s10_kv_cache.py

echo "== Supplementary Table 11 (reasoning effort of gpt-oss-20b) =="
$PY analysis/humaneval/table_s11_reasoning_effort_humaneval.py
$PY analysis/gsm8k_mmlu/table_s11_reasoning_effort_mmlu.py

echo "Done. Figures: outputs/figures   Tables: outputs/tables"
