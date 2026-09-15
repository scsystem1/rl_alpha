#!/usr/bin/env bash
set -uo pipefail

# Run the two native recent-alpha baselines together. The primary placement
# deliberately tests six concurrent processes: three QuantEvolver seeds on
# the free A100 (cuda:2), and three small AlphaSAGE seeds on cuda:3 alongside
# the already-running main GRPO service. Failed cells retry on cuda:0/1/4.

bundle_id="${1:-recent_alpha_baselines_cuda23_20260912}"
qe_config="${2:-configs/experiment/recent_alpha_quantevolver_rolling.yaml}"
as_config="${3:-configs/experiment/recent_alpha_alphasage_rolling.yaml}"
python_qe="${RLALPHA_QE_PYTHON:-/home/sunyuxiang/rl_alpha/.venvs/quantevolver/bin/python}"
python_as="${RLALPHA_AS_PYTHON:-/home/sunyuxiang/rl_alpha/.venvs/alphasage/bin/python}"
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs_root="/data/sunyuxiang/rl_alpha/runs"
qe_id="${bundle_id}/quantevolver"
as_id="${bundle_id}/alphasage"
bundle_root="${runs_root}/${bundle_id}"
max_attempts="${RLALPHA_BASELINE_MAX_ATTEMPTS:-2}"

cd "${code_root}"

if [[ ! -x "${python_qe}" || ! -x "${python_as}" ]]; then
  echo "baseline Python environment is unavailable" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "NVIDIA driver/GPU is unavailable" >&2
  exit 3
fi

validate_config() {
  local python_bin="$1" config="$2" expected_method="$3" expected_reward="$4"
  PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" - "${config}" "${expected_method}" "${expected_reward}" <<'PY'
import sys
from rlalpha.config import load_yaml
from rlalpha.rolling import expected_cells

raw = load_yaml(sys.argv[1])
method, reward = sys.argv[2:]
if [int(year) for year in raw["rolling"]["test_years"]] != [2021, 2022, 2023, 2024, 2025]:
    raise SystemExit("baseline configs must contain five windows 2021-2025")
if int(raw["experiment"].get("search_steps", -1)) != 100:
    raise SystemExit("baseline configs must use 100 search steps")
if int(raw["experiment"].get("proposal_group_size", -1)) != 8:
    raise SystemExit("baseline configs must use proposal_group_size=8")
if expected_cells(raw) != [(method, reward, 0), (method, reward, 1), (method, reward, 2)]:
    raise SystemExit(f"unexpected cells for {method}: {expected_cells(raw)!r}")
print("ok")
PY
}

validate_config "${python_qe}" "${qe_config}" quantevolver qe_native >/dev/null || exit 2
validate_config "${python_as}" "${as_config}" alphasage r0 >/dev/null || exit 2

mkdir -p "${bundle_root}"
exec 9>"${bundle_root}/launcher.lock"
if ! flock -n 9; then
  echo "another baseline launcher owns ${bundle_root}/launcher.lock" >&2
  exit 2
fi
exec > >(tee -a "${bundle_root}/launcher.log") 2>&1

timestamp() { date -Iseconds; }

qe_specs=$(PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" "${python_qe}" - "${qe_config}" "${qe_id}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from rlalpha.rolling import prepare_rolling_windows
for year, child_config, child_id in prepare_rolling_windows(sys.argv[1], sys.argv[2]):
    print(f"{year}\t{child_config}\t{child_id}")
PY
)
as_specs=$(PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" "${python_as}" - "${as_config}" "${as_id}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from rlalpha.rolling import prepare_rolling_windows
for year, child_config, child_id in prepare_rolling_windows(sys.argv[1], sys.argv[2]):
    print(f"{year}\t{child_config}\t{child_id}")
PY
)
if [[ -z "${qe_specs}" || -z "${as_specs}" ]]; then
  echo "failed to freeze baseline rolling windows" >&2
  exit 2
fi

declare -A qe_child_config=()
declare -A as_child_config=()
while IFS=$'\t' read -r year child_config child_id; do
  [[ -n "${year}" ]] && qe_child_config["${year}"]="${child_config}"
done <<< "${qe_specs}"
while IFS=$'\t' read -r year child_config child_id; do
  [[ -n "${year}" ]] && as_child_config["${year}"]="${child_config}"
done <<< "${as_specs}"

search_complete() {
  local root="$1" method="$2" reward="$3" seed="$4"
  local cell="${root}/${method}/${reward}/seed_${seed}"
  [[ -f "${cell}/final_pool.json" && -f "${cell}/train_metrics.json" && -f "${cell}/manifest.yaml" ]] || return 1
  "${python_qe}" - "${cell}/train_metrics.json" <<'PY'
import json
import sys
from pathlib import Path
metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if int(metrics.get("completed_steps", -1)) >= 100 else 1)
PY
}

fallback_gpu() {
  local seed="$1"
  case "${seed}" in
    0) echo 0 ;;
    1) echo 1 ;;
    *) echo 4 ;;
  esac
}

run_cell() {
  local year="$1" method="$2" reward="$3" seed="$4" initial_gpu="$5" child_config="$6" child_id="$7" root="$8"
  local cell="${root}/test_${year}/${method}/${reward}/seed_${seed}"
  local scheduler_dir="${cell}/scheduler"
  mkdir -p "${scheduler_dir}"
  if search_complete "${root}/test_${year}" "${method}" "${reward}" "${seed}"; then
    echo "$(timestamp) skip completed ${child_id}/${method}/${reward}/seed_${seed}"
    return 0
  fi

  local attempt rc target_gpu log_file ray_root
  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    target_gpu="${initial_gpu}"
    if (( attempt > 1 )); then
      target_gpu="$(fallback_gpu "${seed}")"
    fi
    log_file="${scheduler_dir}/search_attempt_${attempt}_cuda${target_gpu}.log"
    echo "$(timestamp) start ${child_id}/${method}/${reward}/seed_${seed} attempt=${attempt} gpu=${target_gpu}"
    if [[ "${method}" == "quantevolver" ]]; then
      ray_root="/tmp/rlalpha_qe_${year}_s${seed}_g${target_gpu}_a${attempt}"
      mkdir -p "${ray_root}"
      CUDA_VISIBLE_DEVICES="${target_gpu}" RLALPHA_PHYSICAL_GPU="${target_gpu}" \
      RLALPHA_VLLM_MEMORY_UTILIZATION=0.12 RLALPHA_GRPO_MICROBATCH=1 RAY_TMPDIR="${ray_root}" \
      PYTHONPATH="${code_root}/src:/home/sunyuxiang/rl_alpha/baseline/QuantEvolver${PYTHONPATH:+:${PYTHONPATH}}" \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
      "${python_qe}" -u -m rlalpha.cli search run \
        --method quantevolver --reward qe_native --seed "${seed}" --steps 100 \
        --experiment-id "${child_id}" --config "${child_config}" >>"${log_file}" 2>&1
      rc=$?
    else
      CUDA_VISIBLE_DEVICES="${target_gpu}" RLALPHA_PHYSICAL_GPU="${target_gpu}" \
      RLALPHA_ALPHASAGE_ROOT=/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE \
      PYTHONPATH="${code_root}/src:/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE:/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE/src${PYTHONPATH:+:${PYTHONPATH}}" \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
      "${python_as}" -u -m rlalpha.cli search run \
        --method alphasage --reward r0 --seed "${seed}" --steps 100 \
        --experiment-id "${child_id}" --config "${child_config}" >>"${log_file}" 2>&1
      rc=$?
    fi
    if [[ ${rc} -eq 0 ]] && search_complete "${root}/test_${year}" "${method}" "${reward}" "${seed}"; then
      echo "$(timestamp) search complete ${child_id}/${method}/${reward}/seed_${seed} attempt=${attempt}"
      return 0
    fi
    if search_complete "${root}/test_${year}" "${method}" "${reward}" "${seed}"; then
      echo "$(timestamp) artifacts complete despite rc=${rc}: ${child_id}/${method}/${reward}/seed_${seed}"
      return 0
    fi
    echo "$(timestamp) failed ${child_id}/${method}/${reward}/seed_${seed} attempt=${attempt} rc=${rc}; retrying on fallback GPU"
    sleep 15
  done
  echo "$(timestamp) exhausted attempts: ${child_id}/${method}/${reward}/seed_${seed}" >&2
  return 1
}

run_window() {
  local year="$1"
  local -a worker_pids=()
  local -A worker_keys=()
  start_cell() {
    local method="$1" reward="$2" seed="$3" initial_gpu="$4" child_config="$5" child_id="$6" root="$7"
    run_cell "${year}" "${method}" "${reward}" "${seed}" "${initial_gpu}" "${child_config}" "${child_id}" "${root}" &
    local pid=$!
    worker_pids+=("${pid}")
    worker_keys["${pid}"]="${child_id}/${method}/${reward}/seed_${seed}"
  }

  echo "$(timestamp) begin baseline window test_${year}; launching six cells together"
  for seed in 0 1 2; do
    start_cell quantevolver qe_native "${seed}" 2 "${qe_child_config[${year}]}" "${qe_id}/test_${year}" "${runs_root}/${qe_id}"
  done
  for seed in 0 1 2; do
    start_cell alphasage r0 "${seed}" 3 "${as_child_config[${year}]}" "${as_id}/test_${year}" "${runs_root}/${as_id}"
  done

  local failures=0 pid
  for pid in "${worker_pids[@]}"; do
    if wait "${pid}"; then
      echo "$(timestamp) worker succeeded ${worker_keys[${pid}]}"
    else
      echo "$(timestamp) worker failed ${worker_keys[${pid}]}" >&2
      failures=$((failures + 1))
    fi
  done
  if (( failures > 0 )); then
    echo "$(timestamp) baseline window test_${year} failed; stopping before next window" >&2
    return 1
  fi
  echo "$(timestamp) complete baseline window test_${year}; six cells reached 100 steps"
}

echo "$(timestamp) baseline bundle=${bundle_id}; QE=${qe_id}; AlphaSAGE=${as_id}"
echo "$(timestamp) primary placement: QE seeds 0/1/2 -> cuda:2; AlphaSAGE seeds 0/1/2 -> cuda:3"
for year in 2021 2022 2023 2024 2025; do
  run_window "${year}" || exit 1
done

echo "$(timestamp) all baseline windows frozen; starting evaluation/report"
PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1 \
"${python_qe}" -u - "${qe_config}" "${qe_id}" "${as_config}" "${as_id}" <<'PY' >>"${bundle_root}/evaluation.log" 2>&1
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from rlalpha.rolling import evaluate_rolling
from rlalpha.reporting.build import build_report

qe_config, qe_id, as_config, as_id = sys.argv[1:]
for config, experiment_id in ((qe_config, qe_id), (as_config, as_id)):
    evaluation = evaluate_rolling(experiment_id, config)
    print(json.dumps(evaluation, indent=2, default=str))
    if not evaluation.get("complete"):
        raise SystemExit(f"incomplete rolling evaluation: {experiment_id}")
    report = build_report(experiment_id, config)
    print(json.dumps(report, indent=2, default=str))
PY
rc=$?
if [[ ${rc} -ne 0 ]]; then
  echo "$(timestamp) baseline evaluation/report failed rc=${rc}; see ${bundle_root}/evaluation.log" >&2
  exit "${rc}"
fi
echo "$(timestamp) QuantEvolver and AlphaSAGE baseline bundle complete"
