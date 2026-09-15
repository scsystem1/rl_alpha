#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 EXPERIMENT_ID CONFIG [WORKERS]" >&2
  exit 2
fi

experiment_id="$1"
config="$2"
workers="${3:-2}"
python_bin="${RLALPHA_PYTHON:-/home/sunyuxiang/miniconda3/envs/rlalpha/bin/python}"
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"
steps="250"

cd "${code_root}"
while true; do
  if [[ -f "${run_root}/evaluation_status.json" ]] && jq -e '.status == "complete"' "${run_root}/evaluation_status.json" >/dev/null 2>&1; then
    exit 0
  fi

  ready=1
  for method in random gp base_llm grpo_llm; do
    for seed in 0 1 2; do
      cell="${run_root}/${method}/r1_oof/seed_${seed}"
      if [[ ! -f "${cell}/final_pool.json" || ! -f "${cell}/train_metrics.json" || ! -f "${cell}/manifest.yaml" ]]; then
        ready=0
        continue
      fi
      if ! jq -e --argjson steps "${steps}" '.completed_steps >= $steps' "${cell}/train_metrics.json" >/dev/null 2>&1; then
        ready=0
      fi
    done
  done

  if [[ ${ready} -eq 1 ]]; then
    echo "$(date -Iseconds) all 12 searches accepted; starting evaluation"
    CUDA_VISIBLE_DEVICES="" \
    OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
    PYTHONUNBUFFERED=1 "${python_bin}" -u scripts/evaluate_r1_oof_experiment.py \
      --config "${config}" --experiment-id "${experiment_id}" --workers "${workers}"
    exit $?
  fi
  sleep 60
done
