#!/usr/bin/env bash
set -uo pipefail

experiment_id="${1:-r1_oof_4methods_seed34_20260907}"
config="${2:-configs/experiment/r1_oof_4methods_seed34.yaml}"
reward="${3:-r1_oof}"
python_bin="${RLALPHA_PYTHON:-/home/sunyuxiang/miniconda3/envs/rlalpha/bin/python}"
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"
max_attempts="${RLALPHA_MAX_ATTEMPTS:-3}"
evaluation_workers="${RLALPHA_EVALUATION_WORKERS:-2}"

cd "${code_root}"
"${python_bin}" -c '
import sys, yaml
config, expected = sys.argv[1:]
experiment = (yaml.safe_load(open(config, encoding="utf-8")) or {})["experiment"]
seeds = [int(seed) for seed in experiment["seeds"]]
if seeds != [3, 4]:
    raise SystemExit(f"this launcher requires seeds [3, 4], got {seeds!r}")
rewards = {str(cell[1]) for cell in experiment["cells"]}
if rewards != {expected}:
    raise SystemExit(f"launcher reward {expected!r} does not match configured rewards {sorted(rewards)!r}")
' "${config}" "${reward}"
mkdir -p "${run_root}"
exec 9>"${run_root}/launcher.lock"
if ! flock -n 9; then
  echo "another launcher already owns ${run_root}/launcher.lock" >&2
  exit 2
fi
exec > >(tee -a "${run_root}/launcher.log") 2>&1

timestamp() { date -Iseconds; }

search_complete() {
  local method="$1" seed="$2"
  local cell="${run_root}/${method}/${reward}/seed_${seed}"
  [[ -f "${cell}/final_pool.json" && -f "${cell}/train_metrics.json" && -f "${cell}/manifest.yaml" ]] || return 1
  jq -e --argjson steps 250 '.completed_steps >= $steps' "${cell}/train_metrics.json" >/dev/null 2>&1
}

run_cell() {
  local method="$1" seed="$2" gpu="$3"
  local key="${method}/${reward}/seed_${seed}"
  local scheduler_dir="${run_root}/${key}/scheduler"
  mkdir -p "${scheduler_dir}"
  if search_complete "${method}" "${seed}"; then
    echo "$(timestamp) skip completed search ${key}"
    return 0
  fi
  local attempt rc log_file ray_root vllm_utilization
  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    log_file="${scheduler_dir}/search_attempt_${attempt}.log"
    echo "$(timestamp) start search ${key} attempt=${attempt} gpu=${gpu:-cpu} log=${log_file}"
    if [[ -z "${gpu}" ]]; then
      CUDA_VISIBLE_DEVICES="" \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
      "${python_bin}" -u -m rlalpha.cli search run \
        --method "${method}" --reward "${reward}" --seed "${seed}" --steps 250 \
        --experiment-id "${experiment_id}" --config "${config}" >>"${log_file}" 2>&1
      rc=$?
    else
      ray_root="/tmp/oof_s34_g${gpu}_${method}_s${seed}_a${attempt}"
      mkdir -p "${ray_root}"
      vllm_utilization="0.12"
      if [[ "${gpu}" == "0" ]]; then
        vllm_utilization="0.16"
      fi
      if [[ "${method}" == "grpo_llm" ]]; then
        CUDA_VISIBLE_DEVICES="${gpu}" RLALPHA_PHYSICAL_GPU="${gpu}" \
        RLALPHA_VLLM_MEMORY_UTILIZATION="${vllm_utilization}" RLALPHA_GRPO_MICROBATCH=1 \
        RLALPHA_R1_OOF_RUNTIME_HOOKS=1 RAY_TMPDIR="${ray_root}" \
        PYTHONPATH="${code_root}/scripts/r1_oof_runtime:${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
        "${python_bin}" -u -m rlalpha.cli search run \
          --method "${method}" --reward "${reward}" --seed "${seed}" --steps 250 \
          --experiment-id "${experiment_id}" --config "${config}" >>"${log_file}" 2>&1
        rc=$?
      else
        CUDA_VISIBLE_DEVICES="${gpu}" RLALPHA_PHYSICAL_GPU="${gpu}" \
        RLALPHA_VLLM_MEMORY_UTILIZATION="${vllm_utilization}" \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
        "${python_bin}" -u -m rlalpha.cli search run \
          --method "${method}" --reward "${reward}" --seed "${seed}" --steps 250 \
          --experiment-id "${experiment_id}" --config "${config}" >>"${log_file}" 2>&1
        rc=$?
      fi
    fi
    if [[ ${rc} -eq 0 ]] && search_complete "${method}" "${seed}"; then
      echo "$(timestamp) search complete ${key} attempt=${attempt}"
      return 0
    fi
    if search_complete "${method}" "${seed}"; then
      echo "$(timestamp) search artifacts complete despite rc=${rc}: ${key}"
      return 0
    fi
    echo "$(timestamp) search failed ${key} attempt=${attempt} rc=${rc}; retrying from checkpoint"
    sleep 30
  done
  echo "$(timestamp) exhausted ${max_attempts} attempts: ${key}" >&2
  return 1
}

declare -a worker_pids=()
declare -A worker_keys=()

start_cell() {
  local method="$1" seed="$2" gpu="$3"
  run_cell "${method}" "${seed}" "${gpu}" &
  local pid=$!
  worker_pids+=("${pid}")
  worker_keys["${pid}"]="${method}/${reward}/seed_${seed}"
}

echo "$(timestamp) experiment=${experiment_id} reward=${reward} config=${config}"
echo "$(timestamp) launching all eight searches concurrently"
for method in random gp; do
  for seed in 3 4; do
    start_cell "${method}" "${seed}" ""
  done
done
start_cell base_llm 3 0
start_cell grpo_llm 3 "${RLALPHA_GRPO3_GPU:-1}"
start_cell base_llm 4 3
start_cell grpo_llm 4 3

failures=0
for pid in "${worker_pids[@]}"; do
  if wait "${pid}"; then
    echo "$(timestamp) worker succeeded ${worker_keys[${pid}]}"
  else
    echo "$(timestamp) worker failed ${worker_keys[${pid}]}" >&2
    failures=$((failures + 1))
  fi
done
if (( failures > 0 )); then
  echo "$(timestamp) ${failures} search workers failed; evaluation was not opened" >&2
  exit 1
fi

echo "$(timestamp) all 8 searches frozen; starting standard evaluation with ${evaluation_workers} workers"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
PYTHONUNBUFFERED=1 "${python_bin}" -u scripts/evaluate_r1_oof_experiment.py \
  --config "${config}" --experiment-id "${experiment_id}" --workers "${evaluation_workers}" \
  >>"${run_root}/evaluation.log" 2>&1
rc=$?
if [[ ${rc} -ne 0 ]]; then
  echo "$(timestamp) evaluation failed rc=${rc}; see ${run_root}/evaluation.log" >&2
  exit "${rc}"
fi
echo "$(timestamp) all searches, evaluations, and report generation complete"
