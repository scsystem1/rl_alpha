#!/usr/bin/env bash
set -uo pipefail

experiment_id="${1:-recent_alpha_rolling_cuda0123_20260912}"
config="${2:-configs/experiment/recent_alpha_rolling_cuda0123.yaml}"
python_bin="${RLALPHA_PYTHON:-/home/sunyuxiang/miniconda3/envs/rlalpha/bin/python}"
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"
max_attempts="${RLALPHA_MAX_ATTEMPTS:-3}"
evaluation_workers="${RLALPHA_EVALUATION_WORKERS:-2}"

cd "${code_root}"

if [[ ! -x "${python_bin}" ]]; then
  echo "Python interpreter not found or not executable: ${python_bin}" >&2
  exit 2
fi

config_values=$(PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" - "${config}" <<'PY'
import sys
from pathlib import Path
import yaml

config_path = Path(sys.argv[1])
raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
rolling = raw.get("rolling", {})
experiment = raw.get("experiment", {})
if [int(year) for year in rolling.get("test_years", [])] != [2021, 2022, 2023, 2024, 2025]:
    raise SystemExit(f"unexpected rolling test years: {rolling.get('test_years')!r}")
if experiment.get("methods") != ["random", "gp", "base_llm", "grpo_llm"]:
    raise SystemExit(f"unexpected methods: {experiment.get('methods')!r}")
if experiment.get("rewards") != ["r1_oof"] or [int(seed) for seed in experiment.get("seeds", [])] != [0, 1, 2]:
    raise SystemExit("rolling launcher requires r1_oof and seeds [0, 1, 2]")
if int(experiment.get("search_steps", -1)) != 100:
    raise SystemExit(f"this launcher requires 100 search steps, got {experiment.get('search_steps')!r}")
if int(experiment.get("proposal_group_size", -1)) != 8:
    raise SystemExit("recent-alpha fairness protocol requires proposal_group_size=8")
if list(experiment.get("gpu_devices", {}).get("base_llm", [])) != [1, 2]:
    raise SystemExit("base_llm GPU metadata must be [1, 2]")
if list(experiment.get("gpu_devices", {}).get("grpo_llm", [])) != [0, 1, 3]:
    raise SystemExit("grpo_llm GPU metadata must be [0, 1, 3]")
if not bool(experiment.get("auto_start_expensive_jobs", False)):
    raise SystemExit("expensive jobs are disabled; refusing to start")
paths = raw.get("paths", {})
print(paths.get("runs_root", "/data/sunyuxiang/rl_alpha/runs"))
PY
)
if [[ $? -ne 0 ]]; then
  echo "configuration validation failed" >&2
  exit 2
fi
run_root="${config_values}/${experiment_id}"

if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "NVIDIA driver/GPU is unavailable; refusing to start the CUDA topology" >&2
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

window_specs=$(PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
  "${python_bin}" - "${config}" "${experiment_id}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from rlalpha.rolling import prepare_rolling_windows

for year, child_config, child_id in prepare_rolling_windows(sys.argv[1], sys.argv[2]):
    print(f"{year}\t{child_config}\t{child_id}")
PY
)
if [[ $? -ne 0 ]]; then
  echo "failed to freeze rolling window configurations" >&2
  exit 2
fi

search_complete() {
  local child_root="$1" method="$2" seed="$3"
  local cell="${child_root}/${method}/r1_oof/seed_${seed}"
  [[ -f "${cell}/final_pool.json" && -f "${cell}/train_metrics.json" && -f "${cell}/manifest.yaml" ]] || return 1
  "${python_bin}" - "${cell}/train_metrics.json" <<'PY'
import json
import sys
from pathlib import Path
metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if int(metrics.get("completed_steps", -1)) >= 100 else 1)
PY
}

run_cell() {
  local year="$1" child_config="$2" child_id="$3" method="$4" seed="$5" gpu="$6"
  local child_root="${run_root}/test_${year}"
  local key="${method}/r1_oof/seed_${seed}"
  local scheduler_dir="${child_root}/${key}/scheduler"
  mkdir -p "${scheduler_dir}"
  if search_complete "${child_root}" "${method}" "${seed}"; then
    echo "$(timestamp) skip completed test_${year}/${key}"
    return 0
  fi

  local attempt rc log_file ray_root target_gpu vllm_utilization
  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    target_gpu="${gpu}"
    # The first placement tests the requested H800 slot. If it cannot create
    # the GRPO cache, retry this cell on the known-capable A100 slot.
    if [[ "${method}" == "grpo_llm" && "${seed}" == "1" && ${attempt} -gt 1 ]]; then
      target_gpu="2"
    fi
    log_file="${scheduler_dir}/search_attempt_${attempt}.log"
    echo "$(timestamp) start test_${year}/${key} attempt=${attempt} gpu=${target_gpu:-cpu} log=${log_file}"
    if [[ -z "${target_gpu}" ]]; then
      CUDA_VISIBLE_DEVICES="" \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
      PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_bin}" -u -m rlalpha.cli search run \
        --method "${method}" --reward r1_oof --seed "${seed}" --steps 100 \
        --experiment-id "${child_id}" --config "${child_config}" >>"${log_file}" 2>&1
      rc=$?
    else
      ray_root="/tmp/rlalpha_recent_${year}_g${target_gpu}_s${seed}_a${attempt}"
      mkdir -p "${ray_root}"
      if [[ "${method}" == "grpo_llm" ]]; then
        vllm_utilization="0.12"
        CUDA_VISIBLE_DEVICES="${target_gpu}" RLALPHA_PHYSICAL_GPU="${target_gpu}" \
        RLALPHA_VLLM_MEMORY_UTILIZATION="${vllm_utilization}" RLALPHA_GRPO_MICROBATCH=1 \
        RLALPHA_R1_OOF_RUNTIME_HOOKS=1 RAY_TMPDIR="${ray_root}" \
        PYTHONPATH="${code_root}/scripts/r1_oof_runtime:${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
        "${python_bin}" -u -m rlalpha.cli search run \
          --method "${method}" --reward r1_oof --seed "${seed}" --steps 100 \
          --experiment-id "${child_id}" --config "${child_config}" >>"${log_file}" 2>&1
        rc=$?
      else
        if [[ "${target_gpu}" == "1" ]]; then
          # RTX 4090 has only 48 GiB and shares this slot with a GRPO actor;
          # 0.16 is below the vLLM weight/activation/non-torch floor there.
          vllm_utilization="0.40"
        else
          vllm_utilization="0.16"
        fi
        CUDA_VISIBLE_DEVICES="${target_gpu}" RLALPHA_PHYSICAL_GPU="${target_gpu}" \
        RLALPHA_VLLM_MEMORY_UTILIZATION="${vllm_utilization}" \
        PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
        OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
        HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
        "${python_bin}" -u -m rlalpha.cli search run \
          --method "${method}" --reward r1_oof --seed "${seed}" --steps 100 \
          --experiment-id "${child_id}" --config "${child_config}" >>"${log_file}" 2>&1
        rc=$?
      fi
    fi
    if [[ ${rc} -eq 0 ]] && search_complete "${child_root}" "${method}" "${seed}"; then
      echo "$(timestamp) search complete test_${year}/${key} attempt=${attempt}"
      return 0
    fi
    if search_complete "${child_root}" "${method}" "${seed}"; then
      echo "$(timestamp) search artifacts complete despite rc=${rc}: test_${year}/${key}"
      return 0
    fi
    echo "$(timestamp) search failed test_${year}/${key} attempt=${attempt} rc=${rc}; retrying"
    sleep 15
  done
  echo "$(timestamp) exhausted ${max_attempts} attempts: test_${year}/${key}" >&2
  return 1
}

run_window() {
  local year="$1" child_config="$2" child_id="$3"
  local child_root="${run_root}/test_${year}"
  local -a worker_pids=()
  local -A worker_keys=()

  start_cell() {
    local method="$1" seed="$2" gpu="$3"
    run_cell "${year}" "${child_config}" "${child_id}" "${method}" "${seed}" "${gpu}" &
    local pid=$!
    worker_pids+=("${pid}")
    worker_keys["${pid}"]="${method}/r1_oof/seed_${seed}"
  }

  echo "$(timestamp) begin window test_${year}; launching all 12 cells"
  for method in random gp; do
    for seed in 0 1 2; do
      start_cell "${method}" "${seed}" ""
    done
  done
  start_cell grpo_llm 0 0
  start_cell grpo_llm 2 1
  start_cell grpo_llm 1 3
  start_cell base_llm 1 2
  start_cell base_llm 2 2

  # cuda:1 is shared by GRPO seed 2 and Base-LLM seed 0. Let the GRPO
  # process finish vLLM startup first; otherwise both vLLM profilers see a
  # moving memory floor and one of them can reject its KV cache.
  local grpo1_log="${child_root}/grpo_llm/r1_oof/seed_2/scheduler/search_attempt_1.log"
  local ready=0 wait_count
  for ((wait_count=1; wait_count<=90; wait_count++)); do
    if [[ -f "${grpo1_log}" ]] && rg -q 'step:1 -|training/global_step:1' "${grpo1_log}"; then
      ready=1
      break
    fi
    sleep 2
  done
  if (( ready )); then
    echo "$(timestamp) cuda:1 GRPO seed 2 reached first training step; starting Base-LLM seed 0"
  else
    echo "$(timestamp) cuda:1 GRPO seed 2 did not expose a first-step marker within 180s; starting Base-LLM seed 0 and relying on retries"
  fi
  start_cell base_llm 0 1

  local failures=0 pid
  for pid in "${worker_pids[@]}"; do
    if wait "${pid}"; then
      echo "$(timestamp) worker succeeded test_${year}/${worker_keys[${pid}]}"
    else
      echo "$(timestamp) worker failed test_${year}/${worker_keys[${pid}]}" >&2
      failures=$((failures + 1))
    fi
  done
  if (( failures > 0 )); then
    echo "$(timestamp) test_${year}: ${failures} workers failed; stopping before next window" >&2
    return 1
  fi
  echo "$(timestamp) complete window test_${year}; all 12 searches reached 100 steps"
}

echo "$(timestamp) experiment=${experiment_id} config=${config} windows=5 cells=60 steps=100"
echo "$(timestamp) topology: cuda0=grpo0 cuda1=grpo2,base0 cuda2=base1,base2 cuda3=grpo1(cpu fallback=random/gp)"

while IFS=$'\t' read -r year child_config child_id; do
  [[ -n "${year}" ]] || continue
  if ! run_window "${year}" "${child_config}" "${child_id}"; then
    exit 1
  fi
done <<< "${window_specs}"

echo "$(timestamp) all 5 windows and 60 searches frozen; starting rolling finalization"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 NUMBA_NUM_THREADS=4 \
PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1 \
"${python_bin}" -u - <<'PY' "${config}" "${experiment_id}" >>"${run_root}/evaluation.log" 2>&1
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from rlalpha.rolling import evaluate_rolling
from rlalpha.reporting.build import build_report

config, experiment_id = sys.argv[1:]
evaluation = evaluate_rolling(experiment_id, config)
print(json.dumps(evaluation, indent=2, default=str))
if not evaluation.get("complete"):
    raise SystemExit("rolling evaluation is incomplete")
report = build_report(experiment_id, config)
print(json.dumps(report, indent=2, default=str))
PY
rc=$?
if [[ ${rc} -ne 0 ]]; then
  echo "$(timestamp) rolling finalization/report failed rc=${rc}; see ${run_root}/evaluation.log" >&2
  exit "${rc}"
fi
echo "$(timestamp) all 60 searches, rolling evaluations, and report generation complete"
