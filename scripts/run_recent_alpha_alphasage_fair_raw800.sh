#!/usr/bin/env bash
set -euo pipefail

bundle_id="${1:-recent_alpha_alphasage_fair_raw800_cuda2_20260912}"
as_config="${2:-configs/experiment/recent_alpha_alphasage_rolling.yaml}"
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
runs_root="/data/sunyuxiang/rl_alpha/runs"
python_as="${RLALPHA_AS_PYTHON:-/home/sunyuxiang/rl_alpha/.venvs/alphasage/bin/python}"
as_id="${bundle_id}/alphasage"
bundle_root="${runs_root}/${bundle_id}"
as_root="${bundle_root}/alphasage"

cd "${code_root}"
mkdir -p "${bundle_root}"
exec 9>"${bundle_root}/launcher.lock"
flock -n 9 || { echo "another AlphaSAGE launcher owns ${bundle_root}/launcher.lock" >&2; exit 2; }
exec > >(tee -a "${bundle_root}/launcher.log") 2>&1

timestamp() { date -Is; }

specs=$(PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" "${python_as}" - "${as_config}" "${as_id}" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / "src"))
from rlalpha.rolling import prepare_rolling_windows

for year, child_config, child_id in prepare_rolling_windows(sys.argv[1], sys.argv[2]):
    print(f"{year}\t{child_config}\t{child_id}")
PY
)

declare -A child_config=()
while IFS=$'\t' read -r year config child_id; do
  [[ -n "${year}" ]] && child_config["${year}"]="${config}"
done <<< "${specs}"

is_complete() {
  local year="$1" seed="$2"
  local cell="${as_root}/test_${year}/alphasage/r0/seed_${seed}"
  [[ -f "${cell}/final_pool.json" && -f "${cell}/train_metrics.json" && -f "${cell}/manifest.yaml" ]] || return 1
  "${python_as}" - "${cell}/train_metrics.json" <<'PY'
import json
import sys
from pathlib import Path

metrics = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
raise SystemExit(0 if (
    int(metrics.get("completed_steps", -1)) == 100
    and int(metrics.get("raw_proposals", -1)) == 800
    and int(metrics.get("candidates_per_step", -1)) == 8
) else 1)
PY
}

run_cell() {
  local year="$1" seed="$2"
  local cell="${as_root}/test_${year}/alphasage/r0/seed_${seed}"
  local log_file="${cell}/scheduler/search_fair_raw800_cuda2.log"
  if is_complete "${year}" "${seed}"; then
    echo "$(timestamp) skip completed ${as_id}/test_${year}/alphasage/r0/seed_${seed}"
    return 0
  fi
  mkdir -p "${cell}/scheduler"
  echo "$(timestamp) start ${as_id}/test_${year}/alphasage/r0/seed_${seed} gpu=2 budget=800_raw_proposals"
  CUDA_VISIBLE_DEVICES=2 RLALPHA_PHYSICAL_GPU=2 \
  RLALPHA_ALPHASAGE_ROOT=/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE \
  PYTHONPATH="${code_root}/src:/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE:/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE/src${PYTHONPATH:+:${PYTHONPATH}}" \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
  "${python_as}" -u -m rlalpha.cli search run \
    --method alphasage --reward r0 --seed "${seed}" --steps 100 \
    --experiment-id "${as_id}/test_${year}" \
    --config "${child_config[${year}]}" >>"${log_file}" 2>&1
  is_complete "${year}" "${seed}"
  echo "$(timestamp) complete ${as_id}/test_${year}/alphasage/r0/seed_${seed}"
}

echo "$(timestamp) starting AlphaSAGE fair raw-proposal bundle=${bundle_id}"
echo "$(timestamp) protocol=recent_alpha_v1; windows=2021-2025; per cell=100 steps x 8 slots=800 raw proposals; gpu=cuda:2"

for year in 2021 2022 2023 2024 2025; do
  echo "$(timestamp) begin window test_${year}; launching AlphaSAGE seeds 0/1/2 concurrently on cuda:2"
  pids=()
  for seed in 0 1 2; do
    run_cell "${year}" "${seed}" &
    pids+=("$!")
  done
  failures=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failures=$((failures + 1))
    fi
  done
  if (( failures > 0 )); then
    echo "$(timestamp) window test_${year} failed; stopping before next window" >&2
    exit 1
  fi
  echo "$(timestamp) complete window test_${year}"
done

echo "$(timestamp) all AlphaSAGE fair raw-proposal cells complete"
