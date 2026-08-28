#!/usr/bin/env bash
set -euo pipefail

experiment_id="${1:-revision_v3_full_pilot_cuda3_20260811}"
config="configs/experiment/revision_v3_full_pilot_cuda3.yaml"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"

mkdir -p "${run_root}"
exec > >(tee -a "${run_root}/launcher.log") 2>&1

date -Iseconds
echo "Starting ${experiment_id} with ${config}"
python -u -m rlalpha.cli matrix run --config "${config}" --experiment-id "${experiment_id}"
python -u -m rlalpha.cli evaluate run --config "${config}" --experiment-id "${experiment_id}"
python -u -m rlalpha.cli report build --config "${config}" --experiment-id "${experiment_id}"
echo "Completed ${experiment_id}"
date -Iseconds
