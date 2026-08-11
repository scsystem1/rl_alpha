#!/usr/bin/env bash
set -euo pipefail

experiment_id="${1:-revision_v3_full_3000_cuda23_20260811}"
config="configs/experiment/revision_v3_full_3000_cuda23.yaml"
run_root="artifacts/runs_v3/${experiment_id}"

mkdir -p "${run_root}"
exec > >(tee -a "${run_root}/launcher.log") 2>&1

date -Iseconds
echo "Starting ${experiment_id} with ${config}"
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

if [[ ! -f artifacts/processed_v3/panel/build_manifest.yaml ]]; then
  echo "Building the missing revision-v3 panel"
  python -u -m rlalpha.cli data build --config "${config}"
fi
if [[ ! -f artifacts/processed_v3/panel/risk_build_manifest.yaml ]]; then
  echo "Building the missing revision-v3 risk panel"
  python -u -m rlalpha.cli risk build --config "${config}"
fi
python -c 'import yaml; from pathlib import Path; root=Path("artifacts/processed_v3/panel"); panel=yaml.safe_load((root/"build_manifest.yaml").read_text()); risk=yaml.safe_load((root/"risk_build_manifest.yaml").read_text()); assert panel["build_fingerprint"] == "408f54f547ef553f03e3046becdff785222465f60af38ef901397e352ce9e6b2", panel["build_fingerprint"]; assert risk["build_fingerprint"] == "1a0fa685a8f846a4cfc3373d7a7f77b772c787664dfef04739079a3fba109cda", risk["build_fingerprint"]; assert risk["hard_gates"]["rank_failures"] == 0; assert risk["hard_gates"]["condition_failures"] == 0; assert risk["formal_member_complete_rate"] >= 0.999; print("Verified current revision-v3 panel/risk fingerprints and hard gates")'
python -u -m rlalpha.cli matrix run --config "${config}" --experiment-id "${experiment_id}"
python -u -m rlalpha.cli evaluate run --config "${config}" --experiment-id "${experiment_id}"
python -u -m rlalpha.cli report build --config "${config}" --experiment-id "${experiment_id}"
echo "Completed ${experiment_id}"
date -Iseconds
