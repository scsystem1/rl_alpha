#!/usr/bin/env bash
set -uo pipefail

experiment_id="${1:-recent_alpha_calendar2y_trainfit_200step_grpo_r2_all_cuda01234_20260913}"
config="${2:-configs/experiment/recent_alpha_calendar2y_trainfit_200step_all_cuda01234.yaml}"
python_bin="${RLALPHA_PYTHON:-/home/sunyuxiang/miniconda3/envs/rlalpha/bin/python}"
python_as="${RLALPHA_AS_PYTHON:-/home/sunyuxiang/rl_alpha/.venvs/alphasage/bin/python}"
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs_root="/data/sunyuxiang/rl_alpha/runs"
run_root="${runs_root}/${experiment_id}"
steps=200
group_size=8
raw_budget=1600
max_attempts="${RLALPHA_MAX_ATTEMPTS:-3}"

cd "${code_root}"

timestamp() { date -Iseconds; }

if [[ ! -x "${python_bin}" ]]; then
  echo "Python interpreter is unavailable: ${python_bin}" >&2
  exit 2
fi
if [[ ! -x "${python_as}" ]]; then
  echo "AlphaSAGE Python interpreter is unavailable: ${python_as}" >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null 2>&1 || ! nvidia-smi -L >/dev/null 2>&1; then
  echo "NVIDIA driver/GPU is unavailable" >&2
  exit 3
fi

PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" - "${config}" <<'PY'
import sys
from rlalpha.config import load_yaml
from rlalpha.rolling import expected_cells, window_config

raw = load_yaml(sys.argv[1])
expected_pairs = {
    ("random", "r1_oof"), ("gp", "r1_oof"),
    ("base_llm", "r1_oof"), ("grpo_llm", "r2_paired_oof"),
    ("alphasage", "r0"), ("quantevolver", "qe_native"),
}
if raw["rolling"].get("window_scheme") != "two_calendar_years_then_test_year":
    raise SystemExit("launcher requires two_calendar_years_then_test_year")
if raw["evaluation"].get("ridge_fit_period") != "train":
    raise SystemExit("launcher requires ridge_fit_period=train")
if raw["rolling"]["test_years"] != [2021, 2022, 2023, 2024, 2025]:
    raise SystemExit("launcher requires test years 2021-2025")
if int(raw["experiment"]["search_steps"]) != 200 or int(raw["experiment"]["proposal_group_size"]) != 8:
    raise SystemExit("launcher requires 200 steps x 8 proposals")
if set(map(tuple, raw["experiment"]["cells"])) != expected_pairs or raw["experiment"]["seeds"] != [0, 1, 2]:
    raise SystemExit(f"unexpected experiment cells: {expected_cells(raw)!r}")
child = window_config(raw, 2021, raw["paths"]["code_root"])
if child["data"]["train"] != ["2019-01-01", "2020-12-31"] or child["data"]["test"] != ["2021-01-01", "2021-12-31"]:
    raise SystemExit("resolved calendar window is incorrect")
if child["reward"]["time_folds"] != [
    {"fit": ["2019-01-01", "2019-06-30"], "score": ["2019-07-01", "2019-12-31"]},
    {"fit": ["2019-07-01", "2019-12-31"], "score": ["2020-01-01", "2020-06-30"]},
    {"fit": ["2020-01-01", "2020-06-30"], "score": ["2020-07-01", "2020-12-31"]},
]:
    raise SystemExit("resolved OOF folds are incorrect")
print("configuration validated")
PY
if [[ $? -ne 0 ]]; then
  exit 2
fi

mkdir -p "${run_root}"
exec 9>"${run_root}/launcher.lock"
if ! flock -n 9; then
  echo "another launcher already owns ${run_root}/launcher.lock" >&2
  exit 2
fi
exec > >(tee -a "${run_root}/launcher.log") 2>&1

window_specs=$(PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" - "${config}" "${experiment_id}" <<'PY'
import sys
from rlalpha.rolling import prepare_rolling_windows
for year, child_config, child_id in prepare_rolling_windows(sys.argv[1], sys.argv[2]):
    print(f"{year}\t{child_config}\t{child_id}")
PY
)
if [[ $? -ne 0 || -z "${window_specs}" ]]; then
  echo "failed to freeze rolling window configurations" >&2
  exit 2
fi

declare -A child_configs=()
declare -A child_ids=()
while IFS=$'\t' read -r year child_config child_id; do
  [[ -n "${year}" ]] || continue
  child_configs["${year}"]="${child_config}"
  child_ids["${year}"]="${child_id}"
done <<< "${window_specs}"

cell_dir() {
  local year="$1" method="$2" reward="$3" seed="$4"
  echo "${run_root}/test_${year}/${method}/${reward}/seed_${seed}"
}

artifacts_complete() {
  local year="$1" method="$2" reward="$3" seed="$4"
  local cell
  cell="$(cell_dir "${year}" "${method}" "${reward}" "${seed}")"
  [[ -f "${cell}/final_pool.json" && -f "${cell}/train_metrics.json" && -f "${cell}/manifest.yaml" ]] || return 1
  "${python_bin}" - "${cell}/train_metrics.json" "${steps}" "${raw_budget}" <<'PY'
import json
import sys
from pathlib import Path
m = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
ok = int(m.get("completed_steps", -1)) >= int(sys.argv[2]) and int(m.get("raw_proposals", -1)) >= int(sys.argv[3])
raise SystemExit(0 if ok else 1)
PY
}

mark_complete() {
  local year="$1" method="$2" reward="$3" seed="$4"
  local cell child_config
  cell="$(cell_dir "${year}" "${method}" "${reward}" "${seed}")"
  child_config="${child_configs[${year}]}"
  PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" "${python_bin}" - \
    "${cell}" "${child_config}" "${method}" "${reward}" "${seed}" "${steps}" <<'PY'
import json
import sys
from pathlib import Path
from rlalpha.config import load_paths
from rlalpha.matrix.runner import _cell_acceptance, _expected_cell_identity
from rlalpha.utils.experiment_log import update_progress

cell, config, method, reward, seed, steps = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3], sys.argv[4], int(sys.argv[5]), int(sys.argv[6])
accepted, reason = _cell_acceptance(cell, steps)
if not accepted:
    raise SystemExit(f"cell acceptance failed: {reason}")
identity = _expected_cell_identity(config.resolve(), load_paths(config), method, reward, seed, steps)
state_path = cell / "progress.json"
state = json.loads(state_path.read_text()) if state_path.exists() else {}
state.update(status="complete", evaluation_status="pending", cell_identity=identity, search_steps=steps)
update_progress(state_path, **state)
PY
}

progress_at_least_one() {
  local year="$1" method="$2" reward="$3" seed="$4"
  local path
  path="$(cell_dir "${year}" "${method}" "${reward}" "${seed}")/progress.json"
  if [[ "${method}" == "grpo_llm" ]]; then
    local metrics
    metrics="$(cell_dir "${year}" "${method}" "${reward}" "${seed}")/grpo_session/verl_metrics.jsonl"
    [[ -s "${metrics}" ]] && return 0
  fi
  [[ -f "${path}" ]] || return 1
  "${python_bin}" - "${path}" <<'PY'
import json
import sys
from pathlib import Path
state = json.loads(Path(sys.argv[1]).read_text())
raise SystemExit(0 if int(state.get("completed_steps", 0)) >= 1 else 1)
PY
}

wait_for_first_step() {
  local year="$1" method="$2" reward="$3" seed="$4"
  local count
  for ((count=1; count<=120; count++)); do
    if progress_at_least_one "${year}" "${method}" "${reward}" "${seed}"; then
      echo "$(timestamp) first step observed: test_${year}/${method}/${reward}/seed_${seed}"
      return 0
    fi
    sleep 2
  done
  echo "$(timestamp) no first-step marker after 240s: test_${year}/${method}/${reward}/seed_${seed}; continuing"
  return 0
}

gpu_free_mib() {
  local gpu="$1"
  nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits | awk -F, -v target="${gpu}" '$1 + 0 == target {gsub(/ /, "", $2); print $2}'
}

run_cell() {
  local year="$1" method="$2" reward="$3" seed="$4" gpu_csv="$5" attempts="${6:-${max_attempts}}"
  local cell scheduler_dir child_config child_id
  cell="$(cell_dir "${year}" "${method}" "${reward}" "${seed}")"
  scheduler_dir="${cell}/scheduler"
  child_config="${child_configs[${year}]}"
  child_id="${child_ids[${year}]}"
  mkdir -p "${scheduler_dir}"
  if artifacts_complete "${year}" "${method}" "${reward}" "${seed}"; then
    mark_complete "${year}" "${method}" "${reward}" "${seed}"
    echo "$(timestamp) skip completed test_${year}/${method}/${reward}/seed_${seed}"
    return 0
  fi

  local -a gpu_choices=()
  if [[ -z "${gpu_csv}" ]]; then
    gpu_choices=("")
  else
    IFS=',' read -r -a gpu_choices <<< "${gpu_csv}"
  fi
  local attempt gpu log_file ray_root rc vllm_util
  for ((attempt=1; attempt<=attempts; attempt++)); do
    gpu="${gpu_choices[$(((attempt - 1) % ${#gpu_choices[@]}))]}"
    log_file="${scheduler_dir}/search_attempt_${attempt}_cuda${gpu:-cpu}.log"
    echo "$(timestamp) start test_${year}/${method}/${reward}/seed_${seed} attempt=${attempt} gpu=${gpu:-cpu} log=${log_file}"
    if [[ -z "${gpu}" ]]; then
      CUDA_VISIBLE_DEVICES="" \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 \
      PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      timeout --signal=TERM --kill-after=2m 270m \
      "${python_bin}" -u -m rlalpha.cli search run \
        --method "${method}" --reward "${reward}" --seed "${seed}" --steps "${steps}" \
        --experiment-id "${child_id}" --config "${child_config}" >>"${log_file}" 2>&1
      rc=$?
    elif [[ "${method}" == "base_llm" ]]; then
      CUDA_VISIBLE_DEVICES="${gpu}" RLALPHA_PHYSICAL_GPU="${gpu}" \
      RLALPHA_VLLM_MEMORY_UTILIZATION=0.18 \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
      PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      timeout --signal=TERM --kill-after=2m 270m \
      "${python_bin}" -u -m rlalpha.cli search run \
        --method "${method}" --reward "${reward}" --seed "${seed}" --steps "${steps}" \
        --experiment-id "${child_id}" --config "${child_config}" >>"${log_file}" 2>&1
      rc=$?
    elif [[ "${method}" == "grpo_llm" ]]; then
      ray_root="/tmp/rlalpha_cal2y_grpo_${year}_s${seed}_g${gpu}_a${attempt}"
      mkdir -p "${ray_root}"
      CUDA_VISIBLE_DEVICES="${gpu}" RLALPHA_PHYSICAL_GPU="${gpu}" \
      RLALPHA_VLLM_MEMORY_UTILIZATION=0.12 RLALPHA_GRPO_MICROBATCH=1 \
      RLALPHA_R1_OOF_RUNTIME_HOOKS=1 RAY_TMPDIR="${ray_root}" \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
      PYTHONPATH="${code_root}/scripts/r1_oof_runtime:${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" \
      timeout --signal=TERM --kill-after=2m 270m \
      "${python_bin}" -u -m rlalpha.cli search run \
        --method "${method}" --reward "${reward}" --seed "${seed}" --steps "${steps}" \
        --experiment-id "${child_id}" --config "${child_config}" >>"${log_file}" 2>&1
      rc=$?
    elif [[ "${method}" == "quantevolver" ]]; then
      ray_root="/tmp/rlalpha_cal2y_qe_${year}_s${seed}_g${gpu}_a${attempt}"
      mkdir -p "${ray_root}"
      vllm_util="${RLALPHA_QE_VLLM_MEMORY_UTILIZATION:-0.09}"
      CUDA_VISIBLE_DEVICES="${gpu}" RLALPHA_PHYSICAL_GPU="${gpu}" \
      RLALPHA_VLLM_MEMORY_UTILIZATION="${vllm_util}" RLALPHA_GRPO_MICROBATCH=1 RAY_TMPDIR="${ray_root}" \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
      HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
      PYTHONPATH="${code_root}/src:/home/sunyuxiang/rl_alpha/baseline/QuantEvolver${PYTHONPATH:+:${PYTHONPATH}}" \
      timeout --signal=TERM --kill-after=2m 270m \
      "${python_bin}" -u -m rlalpha.cli search run \
        --method "${method}" --reward "${reward}" --seed "${seed}" --steps "${steps}" \
        --experiment-id "${child_id}" --config "${child_config}" >>"${log_file}" 2>&1
      rc=$?
    else
      CUDA_VISIBLE_DEVICES="${gpu}" RLALPHA_PHYSICAL_GPU="${gpu}" \
      RLALPHA_ALPHASAGE_ROOT=/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE \
      OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
      PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
      PYTHONPATH="${code_root}/src:/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE:/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE/src${PYTHONPATH:+:${PYTHONPATH}}" \
      "${python_as}" -u -m rlalpha.cli search run \
        --method "${method}" --reward "${reward}" --seed "${seed}" --steps "${steps}" \
        --experiment-id "${child_id}" --config "${child_config}" >>"${log_file}" 2>&1
      rc=$?
    fi
    if artifacts_complete "${year}" "${method}" "${reward}" "${seed}"; then
      if mark_complete "${year}" "${method}" "${reward}" "${seed}"; then
        echo "$(timestamp) complete test_${year}/${method}/${reward}/seed_${seed} attempt=${attempt} rc=${rc}"
        return 0
      fi
    fi
    echo "$(timestamp) failed test_${year}/${method}/${reward}/seed_${seed} attempt=${attempt} gpu=${gpu:-cpu} rc=${rc}"
    sleep 15
  done
  return 1
}

run_window() {
  local year="$1"
  local -a background_pids=() background_keys=() main_pids=() main_keys=()

  start_background() {
    local method="$1" reward="$2" seed="$3" gpus="$4" attempts="${5:-${max_attempts}}"
    run_cell "${year}" "${method}" "${reward}" "${seed}" "${gpus}" "${attempts}" &
    background_pids+=("$!")
    background_keys+=("${method}/${reward}/seed_${seed}")
  }
  start_main() {
    local method="$1" reward="$2" seed="$3" gpus="$4"
    run_cell "${year}" "${method}" "${reward}" "${seed}" "${gpus}" &
    main_pids+=("$!")
    main_keys+=("${method}/${reward}/seed_${seed}")
  }
  echo "$(timestamp) begin test_${year}: 18 cells, 200 steps, 1600 raw proposals each"

  for method in random gp; do
    for seed in 0 1 2; do
      start_background "${method}" r1_oof "${seed}" ""
    done
  done

  start_main grpo_llm r2_paired_oof 0 "2,3,1"
  start_main grpo_llm r2_paired_oof 1 "3,2,1"
  wait_for_first_step "${year}" grpo_llm r2_paired_oof 0
  start_main grpo_llm r2_paired_oof 2 "2,3,1"

  start_main base_llm r1_oof 0 "1,3,2"
  wait_for_first_step "${year}" base_llm r1_oof 0
  start_main base_llm r1_oof 1 "1,3,2"
  wait_for_first_step "${year}" base_llm r1_oof 1
  start_main base_llm r1_oof 2 "1,3,2"

  start_background alphasage r0 0 "0,4,1"
  start_background alphasage r0 1 "4,0,3"
  start_background alphasage r0 2 "4,0,1"

  start_background quantevolver qe_native 0 "0,3,1"
  start_background quantevolver qe_native 1 "3,1,2"
  start_background quantevolver qe_native 2 "1,3,2"
  echo "$(timestamp) all QuantEvolver cells launched: seed_0=cuda:0, seed_1=cuda:3, seed_2=cuda:1"

  local failures=0 i
  for i in "${!main_pids[@]}"; do
    if wait "${main_pids[$i]}"; then
      echo "$(timestamp) main GPU worker succeeded ${main_keys[$i]}"
    else
      echo "$(timestamp) main GPU worker failed ${main_keys[$i]}" >&2
      failures=$((failures + 1))
    fi
  done

  for i in "${!background_pids[@]}"; do
    if wait "${background_pids[$i]}"; then
      echo "$(timestamp) background worker succeeded ${background_keys[$i]}"
    else
      echo "$(timestamp) background worker failed ${background_keys[$i]}" >&2
      failures=$((failures + 1))
    fi
  done

  if (( failures > 0 )); then
    echo "$(timestamp) test_${year} ended with ${failures} failed workers; stopping before the next window" >&2
    return 1
  fi
  echo "$(timestamp) complete test_${year}: all 18 cells accepted at 200 steps / 1600 proposals"
}

echo "$(timestamp) experiment=${experiment_id}"
echo "$(timestamp) protocol=two complete calendar train years -> train ridge fit -> next calendar test year"
echo "$(timestamp) main topology: cuda1=base seeds0/1/2; cuda2=GRPO seeds0/2; cuda3=GRPO seed1; CPU=random+GP"
echo "$(timestamp) baselines: AlphaSAGE on cuda0/4; QuantEvolver seeds0/1/2 on cuda0/3/1"
echo "$(timestamp) initial GPU state:"
nvidia-smi --query-gpu=index,name,memory.used,memory.free,utilization.gpu --format=csv,noheader,nounits

for year in 2021 2022 2023 2024 2025; do
  run_window "${year}" || exit 1
done

echo "$(timestamp) all 90 searches frozen; starting evaluation and rolling report"
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1 \
"${python_bin}" -u - "${config}" "${experiment_id}" <<'PY' >>"${run_root}/evaluation.log" 2>&1
import json
import sys
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
  echo "$(timestamp) evaluation/report failed rc=${rc}; see ${run_root}/evaluation.log" >&2
  exit "${rc}"
fi
echo "$(timestamp) experiment complete"
