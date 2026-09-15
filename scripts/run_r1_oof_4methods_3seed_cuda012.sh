#!/usr/bin/env bash
set -uo pipefail

experiment_id="${1:-r1_oof_4methods_3seed_cuda012_20260912}"
config="${2:-configs/experiment/r1_oof_4methods_3seed_cuda012.yaml}"
reward="${3:-r1_oof}"
python_bin="${RLALPHA_PYTHON:-/home/sunyuxiang/miniconda3/envs/rlalpha/bin/python}"
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"
max_attempts="${RLALPHA_MAX_ATTEMPTS:-3}"
evaluation_workers="${RLALPHA_EVALUATION_WORKERS:-2}"
search_steps=""

cd "${code_root}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python interpreter not found or not executable: ${python_bin}" >&2
  exit 2
fi

config_values=$("${python_bin}" - "${config}" "${reward}" <<'PY'
import sys
from pathlib import Path
import yaml

config_path, expected_reward = sys.argv[1:]
raw = yaml.safe_load(Path(config_path).read_text(encoding="utf-8")) or {}
experiment = raw["experiment"]
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
if int(experiment.get("proposal_group_size", 8)) != 8:
    raise SystemExit("formal four-method runs require proposal_group_size=8")
gpu_devices = experiment.get("gpu_devices", {})
if list(gpu_devices.get("base_llm", [])) != [1, 2]:
    raise SystemExit(f"base_llm GPU metadata must be [1, 2], got {gpu_devices.get('base_llm')!r}")
if list(gpu_devices.get("grpo_llm", [])) != [0, 1, 3]:
    raise SystemExit(f"grpo_llm GPU metadata must be [0, 1, 3], got {gpu_devices.get('grpo_llm')!r}")
paths = raw.get("paths", {})
runs_root = paths.get("runs_root", "/data/sunyuxiang/rl_alpha/runs")
print(int(experiment.get("search_steps", 250)), runs_root)
PY
)
config_status=$?
if [[ ${config_status} -ne 0 ]]; then
  echo "configuration validation failed" >&2
  exit 2
fi
read -r search_steps run_root_from_config <<< "${config_values}"
run_root="${run_root_from_config}/${experiment_id}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "NVIDIA driver/GPU is unavailable; refusing to start the CUDA 0/1/2/3 topology" >&2
  exit 3
fi

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
  "${python_bin}" - "${cell}/train_metrics.json" "${search_steps}" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if int(metrics.get("completed_steps", -1)) >= int(sys.argv[2]) else 1)
PY
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
      # Keep each Ray runtime isolated; GRPO/Base share CUDA 1, while the
      # third GRPO uses CUDA 3 because CUDA 0 cannot hold two GRPO instances.
      ray_root="/tmp/oof012_g${gpu}_s${seed}_a${attempt}"
      mkdir -p "${ray_root}"
      if [[ "${method}" == "grpo_llm" ]]; then
        vllm_utilization="0.12"
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
        vllm_utilization="0.16"
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
echo "$(timestamp) topology: cuda0=grpo0 cuda1=grpo2,base0 cuda2=base1,base2 cuda3=grpo1 cpu=random0-2,gp0-2"

for method in random gp; do
  for seed in 0 1 2; do
    start_cell "${method}" "${seed}" ""
  done
done
start_cell grpo_llm 0 0
start_cell grpo_llm 2 1
start_cell grpo_llm 1 3
start_cell base_llm 0 1
start_cell base_llm 1 2
start_cell base_llm 2 2

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
