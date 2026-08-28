#!/usr/bin/env bash
set -euo pipefail

experiment_id="${1:-revision_v3_full_1000_reward_staged_20260827}"
config="configs/experiment/revision_v3_full_1000_reward_staged.yaml"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"
methods=(random gp base_llm grpo_llm)
rewards=(r1 r0 r2_lcb)

mkdir -p "${run_root}"
exec > >(tee -a "${run_root}/launcher.log") 2>&1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

date -Iseconds
echo "Starting reward-staged experiment ${experiment_id}"
python -c 'import yaml; from pathlib import Path; root=Path("/data/sunyuxiang/rl_alpha/processed/panel"); panel=yaml.safe_load((root/"build_manifest.yaml").read_text()); risk=yaml.safe_load((root/"risk_build_manifest.yaml").read_text()); assert panel["artifact_version"] == 3; assert risk["artifact_version"] == 3; assert risk["hard_gates"]["rank_failures"] == 0; assert risk["hard_gates"]["condition_failures"] == 0; assert risk["formal_member_complete_rate"] >= 0.999; print("Verified panel/risk v3 hard gates")'

for reward in "${rewards[@]}"; do
  echo "Starting reward phase ${reward}"
  matrix_args=()
  for method in "${methods[@]}"; do
    matrix_args+=(--method "${method}")
  done
  python -u -m rlalpha.cli matrix run \
    --config "${config}" \
    --experiment-id "${experiment_id}" \
    --reward "${reward}" \
    "${matrix_args[@]}"
  jq -e --arg marker "/${reward}/" \
    '[.cells | to_entries[] | select(.key | contains($marker)) | .value.status] as $states | ($states | length) == 4 and all($states[]; . == "complete")' \
    "${run_root}/progress.json" >/dev/null
  echo "Completed reward phase ${reward}"
done

python -u -m rlalpha.cli evaluate run --config "${config}" --experiment-id "${experiment_id}"
python -u -m rlalpha.cli report build --config "${config}" --experiment-id "${experiment_id}"
echo "Completed search, evaluation and report for ${experiment_id}"
date -Iseconds
