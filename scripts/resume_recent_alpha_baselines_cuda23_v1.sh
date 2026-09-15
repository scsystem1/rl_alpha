#!/usr/bin/env bash
set -euo pipefail

bundle_id="${1:?usage: $0 BUNDLE_ID [QE_CONFIG] [ALPHASAGE_CONFIG]}"
qe_config="${2:-configs/experiment/recent_alpha_quantevolver_rolling.yaml}"
as_config="${3:-configs/experiment/recent_alpha_alphasage_rolling.yaml}"
code_root="/home/sunyuxiang/rl_alpha/ours"
python_qe="/home/sunyuxiang/rl_alpha/.venvs/quantevolver/bin/python"
python_as="/home/sunyuxiang/rl_alpha/.venvs/alphasage/bin/python"
qe_vllm_util="${RLALPHA_QE_VLLM_MEMORY_UTILIZATION:-0.09}"
bundle_root="/data/sunyuxiang/rl_alpha/runs/${bundle_id}"
qe_id="${bundle_id}/quantevolver"
as_id="${bundle_id}/alphasage"

timestamp() { date -Is; }

is_complete() {
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

wait_complete() {
  local root="$1" method="$2" reward="$3" seed="$4"
  while ! is_complete "${root}" "${method}" "${reward}" "${seed}"; do
    echo "$(timestamp) waiting ${root##*/}/${method}/${reward}/seed_${seed}"
    sleep 30
  done
  echo "$(timestamp) observed complete ${root##*/}/${method}/${reward}/seed_${seed}"
}

run_qe() {
  local year="$1" seed="$2" gpu="$3" attempt="$4"
  local root="${bundle_root}/${qe_id}"
  local cell="${root}/test_${year}/quantevolver/qe_native/seed_${seed}"
  local log_file="${cell}/scheduler/search_attempt_${attempt}_cuda${gpu}.log"
  if is_complete "${root}/test_${year}" quantevolver qe_native "${seed}"; then
    echo "$(timestamp) skip completed ${qe_id}/test_${year}/quantevolver/qe_native/seed_${seed}"
    return 0
  fi
  mkdir -p "${cell}/scheduler"
  echo "$(timestamp) resume QE ${qe_id}/test_${year}/quantevolver/qe_native/seed_${seed} gpu=${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" RLALPHA_PHYSICAL_GPU="${gpu}" \
  RLALPHA_VLLM_MEMORY_UTILIZATION="${qe_vllm_util}" RLALPHA_GRPO_MICROBATCH=1 \
  RAY_TMPDIR="/tmp/rlalpha_qe_${year}_s${seed}_g${gpu}_a${attempt}" \
  PYTHONPATH="${code_root}/src:/home/sunyuxiang/rl_alpha/baseline/QuantEvolver" \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
  "${python_qe}" -u -m rlalpha.cli search run \
    --method quantevolver --reward qe_native --seed "${seed}" --steps 100 \
    --experiment-id "${qe_id}/test_${year}" \
    --config "${bundle_root}/${qe_id}/window_configs/test_${year}.yaml" \
    >>"${log_file}" 2>&1
  is_complete "${root}/test_${year}" quantevolver qe_native "${seed}"
}

run_as() {
  local year="$1" seed="$2"
  local root="${bundle_root}/${as_id}"
  local cell="${root}/test_${year}/alphasage/r0/seed_${seed}"
  local log_file="${cell}/scheduler/search_attempt_resume_cuda3.log"
  if is_complete "${root}/test_${year}" alphasage r0 "${seed}"; then
    echo "$(timestamp) skip completed ${as_id}/test_${year}/alphasage/r0/seed_${seed}"
    return 0
  fi
  mkdir -p "${cell}/scheduler"
  echo "$(timestamp) resume AlphaSAGE ${as_id}/test_${year}/alphasage/r0/seed_${seed} gpu=3"
  CUDA_VISIBLE_DEVICES=3 RLALPHA_PHYSICAL_GPU=3 \
  RLALPHA_ALPHASAGE_ROOT=/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE \
  PYTHONPATH="${code_root}/src:/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE:/home/sunyuxiang/rl_alpha/baseline/AlphaSAGE/src" \
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMBA_NUM_THREADS=8 \
  PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=true \
  "${python_as}" -u -m rlalpha.cli search run \
    --method alphasage --reward r0 --seed "${seed}" --steps 100 \
    --experiment-id "${as_id}/test_${year}" \
    --config "${bundle_root}/${as_id}/window_configs/test_${year}.yaml" \
    >>"${log_file}" 2>&1
  is_complete "${root}/test_${year}" alphasage r0 "${seed}"
}

echo "$(timestamp) resuming bundle=${bundle_id}; QE topology: cuda:0 + cuda:2(2-way); AlphaSAGE: cuda:3"

# The first window already has QE seed0 and AlphaSAGE seeds running from the
# initial six-way probe, while QE seed1 is being recovered separately on cuda:2.
# Wait for those live cells, then run the remaining QE seed2 on cuda:2.
for seed in 0 1 2; do wait_complete "${bundle_root}/${as_id}/test_2021" alphasage r0 "${seed}"; done
wait_complete "${bundle_root}/${qe_id}/test_2021" quantevolver qe_native 0
wait_complete "${bundle_root}/${qe_id}/test_2021" quantevolver qe_native 1
wait_complete "${bundle_root}/${qe_id}/test_2021" quantevolver qe_native 2

for year in 2022 2023 2024 2025; do
  echo "$(timestamp) begin resumed window test_${year}"
  for seed in 0 1 2; do run_as "${year}" "${seed}" & done
  local_as_pids=( $(jobs -rp) )

  run_qe "${year}" 0 0 1 & qe0_pid=$!
  run_qe "${year}" 1 2 1 & qe1_pid=$!
  run_qe "${year}" 2 2 1 & qe2_pid=$!
  wait "${qe0_pid}"
  wait "${qe1_pid}"
  wait "${qe2_pid}"
  for pid in "${local_as_pids[@]}"; do wait "${pid}"; done
  echo "$(timestamp) complete resumed window test_${year}"
done

echo "$(timestamp) resumed windows complete; starting evaluation/report"
PYTHONPATH="${code_root}/src${PYTHONPATH:+:${PYTHONPATH}}" PYTHONUNBUFFERED=1 \
"${python_qe}" -u - "${qe_config}" "${qe_id}" "${as_config}" "${as_id}" <<'PY' \
  >>"${bundle_root}/evaluation_resume.log" 2>&1
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
echo "$(timestamp) resumed baseline bundle complete"
