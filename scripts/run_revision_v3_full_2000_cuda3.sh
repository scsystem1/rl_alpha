#!/usr/bin/env bash
set -euo pipefail

experiment_id="${1:-revision_v3_full_2000_cuda3_20260827}"
config="configs/experiment/revision_v3_full_2000_cuda3.yaml"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"

mkdir -p "${run_root}"
exec > >(tee -a "${run_root}/launcher.log") 2>&1

date -Iseconds
echo "Starting ${experiment_id} with ${config}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export RLALPHA_METHOD_PARALLEL=1

python -c 'import yaml; from pathlib import Path; root=Path("/data/sunyuxiang/rl_alpha/processed/panel"); panel=yaml.safe_load((root/"build_manifest.yaml").read_text()); risk=yaml.safe_load((root/"risk_build_manifest.yaml").read_text()); assert panel["artifact_version"] == 3; assert risk["artifact_version"] == 3; assert risk["hard_gates"]["rank_failures"] == 0; assert risk["hard_gates"]["condition_failures"] == 0; assert risk["formal_member_complete_rate"] >= 0.999; print("Verified panel/risk v3 hard gates")'

scripts/run_experiment.sh "${config}" "${experiment_id}" random gp base_llm grpo_llm
echo "Completed ${experiment_id}"
date -Iseconds
