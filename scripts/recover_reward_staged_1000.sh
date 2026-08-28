#!/usr/bin/env bash
set -euo pipefail

experiment_id="${1:-revision_v3_full_1000_reward_staged_20260827}"
config="configs/experiment/revision_v3_full_1000_reward_staged.yaml"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"
methods=(random gp base_llm grpo_llm)

mkdir -p "${run_root}"
exec > >(tee -a "${run_root}/recovery.log" "${run_root}/launcher.log") 2>&1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "$(date -Iseconds) Waiting for original R1 cells and the parallel GRPO hotfix run"
while tmux has-session -t rlalpha_1000_staged 2>/dev/null \
  || tmux has-session -t rlalpha_grpo_r1_hotfix 2>/dev/null; do
  sleep 30
done

# The hotfix changes only the reward metadata transport: complete diagnostics
# remain in the stage archive while Verl receives a fixed-shape scalar score.
# Re-run its focused tests after the CPU cells release their resources.
timeout 120 /home/sunyuxiang/miniconda3/envs/rlalpha/bin/python -m pytest -q \
  tests/unit/test_verl_grpo_adapter.py \
  tests/unit/test_verl_stage_coordinator.py

python -c '
from pathlib import Path
from rlalpha.matrix.runner import _matrix_progress
root = Path("/data/sunyuxiang/rl_alpha/runs/'"${experiment_id}"'")
_matrix_progress(root, [(method, "r1", 0) for method in ("random", "gp", "base_llm", "grpo_llm")])
'
jq -e '
  [.cells | to_entries[] | select(.key | contains("/r1/")) | .value.status] as $states
  | ($states | length) == 4 and all($states[]; . == "complete")
' "${run_root}/progress.json" >/dev/null
echo "$(date -Iseconds) Completed R1 phase including the parallel repaired GRPO cell"

for reward in r0 r2_lcb; do
  matrix_args=()
  for method in "${methods[@]}"; do
    matrix_args+=(--method "${method}")
  done
  echo "$(date -Iseconds) Starting reward phase ${reward}"
  python -u -m rlalpha.cli matrix run \
    --config "${config}" \
    --experiment-id "${experiment_id}" \
    --reward "${reward}" \
    "${matrix_args[@]}"
  jq -e --arg marker "/${reward}/" '
    [.cells | to_entries[] | select(.key | contains($marker)) | .value.status] as $states
    | ($states | length) == 4 and all($states[]; . == "complete")
  ' "${run_root}/progress.json" >/dev/null
  echo "$(date -Iseconds) Completed reward phase ${reward}"
done

python -u -m rlalpha.cli evaluate run --config "${config}" --experiment-id "${experiment_id}"
python -u -m rlalpha.cli report build --config "${config}" --experiment-id "${experiment_id}"
echo "$(date -Iseconds) Completed recovered experiment and evaluation"
