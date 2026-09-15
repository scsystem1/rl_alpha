#!/usr/bin/env bash
set -uo pipefail

experiment_id="${1:-r2_paired_oof_4methods_3seed_20260908}"
config="${2:-configs/experiment/r2_paired_oof_4methods_3seed.yaml}"
reward="${3:-r2_paired_oof}"
python_bin="${RLALPHA_PYTHON:-/home/sunyuxiang/miniconda3/envs/rlalpha/bin/python}"
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"
max_attempts="${RLALPHA_MAX_ATTEMPTS:-3}"
evaluation_workers="${RLALPHA_EVALUATION_WORKERS:-2}"

cd "${code_root}"
search_steps="$("${python_bin}" -c 'import sys, yaml; print(int((yaml.safe_load(open(sys.argv[1], encoding="utf-8")) or {})["experiment"]["search_steps"]))' "${config}")"
"${python_bin}" - "${config}" "${reward}" <<'PY'
import sys
import yaml

config, expected_reward = sys.argv[1:]
experiment = (yaml.safe_load(open(config, encoding="utf-8")) or {})["experiment"]
cells = [tuple(map(str, cell)) for cell in experiment["cells"]]
expected_cells = [
    ("random", expected_reward),
    ("gp", expected_reward),
    ("base_llm", expected_reward),
    ("grpo_llm", expected_reward),
]
if cells != expected_cells:
    raise SystemExit(f"unexpected experiment cells: {cells!r}")
if [int(seed) for seed in experiment["seeds"]] != [0, 1, 2]:
    raise SystemExit(f"this launcher requires seeds [0, 1, 2], got {experiment['seeds']!r}")
if int(experiment["search_steps"]) != 250:
    raise SystemExit(f"this launcher requires search_steps=250, got {experiment['search_steps']!r}")
if int(experiment.get("proposal_group_size", 8)) != 8:
    raise SystemExit("formal four-method runs require proposal_group_size=8")
PY

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
  jq -e --argjson steps "${search_steps}" '.completed_steps >= $steps' "${cell}/train_metrics.json" >/dev/null 2>&1
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
        --method "${method}" --reward "${reward}" --seed "${seed}" --steps "${search_steps}" \
        --experiment-id "${experiment_id}" --config "${config}" >>"${log_file}" 2>&1
      rc=$?
    else
      # Ray appends a session/socket suffix; keep this prefix short enough for
      # Linux's AF_UNIX 108-byte path limit even with long experiment IDs.
      ray_root="/tmp/r2g${gpu}s${seed}a${attempt}"
      mkdir -p "${ray_root}"
      vllm_utilization="0.12"
      if [[ "${method}" == "base_llm" && ( "${gpu}" == "0" || "${gpu}" == "4" ) ]]; then
        vllm_utilization="0.16"
      elif [[ "${method}" == "base_llm" && "${gpu}" == "1" ]]; then
        # GPU 1 also carries one GRPO actor in the requested topology; keep
        # enough KV cache for base-LLM while staying below the device limit.
        vllm_utilization="0.18"
      elif [[ "${method}" == "grpo_llm" && "${gpu}" == "3" ]]; then
        # A GRPO actor shares GPU 3 with one base-LLM job and resident service.
        vllm_utilization="0.10"
      elif [[ "${method}" == "grpo_llm" && ( "${gpu}" == "0" || "${gpu}" == "4" ) ]]; then
        # GPUs 0 and 4 have resident services; each carries one GRPO only.
        vllm_utilization="0.10"
      fi
      if [[ "${method}" == "grpo_llm" ]]; then
        CUDA_VISIBLE_DEVICES="${gpu}" RLALPHA_PHYSICAL_GPU="${gpu}" \
        RLALPHA_VLLM_MEMORY_UTILIZATION="${vllm_utilization}" RLALPHA_GRPO_MICROBATCH=1 \
        RLALPHA_R1_OOF_RUNTIME_HOOKS=1 RAY_TMPDIR="${ray_root}" \
        PYTHONPATH="${code_root}/scripts/r1_oof_runtime:${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
        "${python_bin}" -u -m rlalpha.cli search run \
          --method "${method}" --reward "${reward}" --seed "${seed}" --steps "${search_steps}" \
          --experiment-id "${experiment_id}" --config "${config}" >>"${log_file}" 2>&1
        rc=$?
      else
        CUDA_VISIBLE_DEVICES="${gpu}" RLALPHA_PHYSICAL_GPU="${gpu}" \
        RLALPHA_VLLM_MEMORY_UTILIZATION="${vllm_utilization}" \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
        "${python_bin}" -u -m rlalpha.cli search run \
          --method "${method}" --reward "${reward}" --seed "${seed}" --steps "${search_steps}" \
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

echo "$(timestamp) experiment=${experiment_id} reward=${reward} steps=${search_steps} config=${config}"
echo "$(timestamp) launching six CPU searches"
for method in random gp; do
  for seed in 0 1 2; do
    start_cell "${method}" "${seed}" ""
  done
done

# Safe initial wave: GPU 0: base-0; GPU 4: base-1;
# GPU 1: base-2 + grpo-0; GPU 3: grpo-1. GPU 3 receives grpo-2 as soon as
# grpo-1 completes; two GRPO actors cannot share that resident-service GPU.
echo "$(timestamp) launching five safe GPU searches; grpo seed 2 is queued on GPU 3"
start_cell base_llm 0 0
start_cell base_llm 1 4
start_cell base_llm 2 1
start_cell grpo_llm 0 1
start_cell grpo_llm 1 3
grpo1_pid="${worker_pids[${#worker_pids[@]}-1]}"

failures=0
if wait "${grpo1_pid}"; then
  echo "$(timestamp) worker succeeded ${worker_keys[${grpo1_pid}]}"
else
  echo "$(timestamp) worker failed ${worker_keys[${grpo1_pid}]}" >&2
  failures=$((failures + 1))
fi
echo "$(timestamp) GPU 3 GRPO slot is available; launching grpo_llm/r2_paired_oof/seed_2"
start_cell grpo_llm 2 3

for pid in "${worker_pids[@]}"; do
  if [[ "${pid}" == "${grpo1_pid}" ]]; then
    continue
  fi
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

echo "$(timestamp) all 12 searches frozen; starting evaluation with ${evaluation_workers} workers"
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
