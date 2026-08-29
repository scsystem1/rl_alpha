#!/usr/bin/env bash
set -euo pipefail

experiment_id="${1:-revision_v3_full_1000_reward_staged_20260827}"
config="configs/experiment/revision_v3_full_1000_reward_staged.yaml"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"
cell="${run_root}/grpo_llm/r1/seed_0"

mkdir -p "${cell}/details"
exec > >(tee -a "${cell}/details/hotfix_launcher.log") 2>&1
export CUDA_VISIBLE_DEVICES=3
export RLALPHA_PHYSICAL_GPU=3
export RLALPHA_VLLM_MEMORY_UTILIZATION=0.15
export RLALPHA_GRPO_MICROBATCH=4
export NUMBA_NUM_THREADS=8
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=8
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

echo "$(date -Iseconds) Starting repaired GRPO R1 directly on physical CUDA:3"
python -u -m rlalpha.cli search run \
  --config "${config}" \
  --experiment-id "${experiment_id}" \
  --method grpo_llm \
  --reward r1 \
  --seed 0 \
  --steps 250

# A direct cell launch deliberately bypasses the matrix lock while the three
# already-running R1 methods remain managed by the original matrix process.
# Perform the same matrix acceptance check and publish the canonical state.
python -c '
from pathlib import Path
from rlalpha.config import load_paths
from rlalpha.matrix.runner import _cell_acceptance, _cell_dir, _expected_cell_identity
from rlalpha.utils.experiment_log import update_progress
config = Path("configs/experiment/revision_v3_full_1000_reward_staged.yaml").resolve()
paths = load_paths(config)
cell = _cell_dir(paths.runs_root / "'"${experiment_id}"'", "grpo_llm", "r1", 0)
accepted, error = _cell_acceptance(cell, 250)
if not accepted:
    raise RuntimeError(error)
identity = _expected_cell_identity(config, paths, "grpo_llm", "r1", 0, 250)
update_progress(cell / "progress.json", status="complete", method="grpo_llm", reward="r1", seed=0, search_steps=250, cell_identity=identity)
'
echo "$(date -Iseconds) Repaired GRPO R1 completed and passed matrix acceptance"
