#!/usr/bin/env bash
set -euo pipefail

experiment_id="${1:-revision_v3_full_3000_cuda23_20260811}"
config="configs/experiment/revision_v3_full_3000_cuda23.yaml"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"

mkdir -p "${run_root}"
exec > >(tee -a "${run_root}/launcher.log") 2>&1

date -Iseconds
echo "Starting ${experiment_id} with ${config}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -f /data/sunyuxiang/rl_alpha/processed/panel/build_manifest.yaml ]]; then
  echo "Building the missing revision-v3 panel"
  python -u -m rlalpha.cli data build --config "${config}"
fi
if [[ ! -f /data/sunyuxiang/rl_alpha/processed/panel/risk_build_manifest.yaml ]]; then
  echo "Building the missing revision-v3 risk panel"
  python -u -m rlalpha.cli risk build --config "${config}"
fi
python -c 'import yaml; from pathlib import Path; root=Path("/data/sunyuxiang/rl_alpha/processed/panel"); panel=yaml.safe_load((root/"build_manifest.yaml").read_text()); risk=yaml.safe_load((root/"risk_build_manifest.yaml").read_text()); assert risk["hard_gates"]["rank_failures"] == 0; assert risk["hard_gates"]["condition_failures"] == 0; assert risk["formal_member_complete_rate"] >= 0.999; print("Verified current panel/risk hard gates")'
python -u -m rlalpha.cli matrix run --config "${config}" --experiment-id "${experiment_id}"
python -u -m rlalpha.cli evaluate run --config "${config}" --experiment-id "${experiment_id}"
python -u -m rlalpha.cli report build --config "${config}" --experiment-id "${experiment_id}"
echo "Completed ${experiment_id}"
date -Iseconds
