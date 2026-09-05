#!/usr/bin/env bash
set -euo pipefail

workspace_root="/home/sunyuxiang/rl_alpha"
python_bin="${workspace_root}/.venvs/quantevolver/bin/python"
cd "${workspace_root}/ours"

smoke_id="${QE_SMOKE_ID:-quantevolver_qwen2b_smoke_cuda4_20260830}"
experiment_id="${QE_EXPERIMENT_ID:-quantevolver_qwen2b_fair250_3seed_cuda4_20260830}"

# The matrix scheduler admits a job above a 20 GiB free-memory floor.
# floor. All smoke and formal seeds run sequentially on physical cuda:4;
# physical cuda:0 is never a scheduling candidate for this experiment.
"${python_bin}" -u -m rlalpha.cli matrix run \
  --config configs/experiment/quantevolver_smoke.yaml \
  --experiment-id "${smoke_id}"

"${python_bin}" -u -m rlalpha.cli matrix run \
  --config configs/experiment/quantevolver_fair_250.yaml \
  --experiment-id "${experiment_id}"

"${python_bin}" -u -m rlalpha.cli evaluate run \
  --config configs/experiment/quantevolver_fair_250.yaml \
  --experiment-id "${experiment_id}" \
  --method quantevolver

"${python_bin}" -u -m rlalpha.cli report build \
  --config configs/experiment/quantevolver_fair_250.yaml \
  --experiment-id "${experiment_id}" \
  --method quantevolver
