#!/usr/bin/env bash
set -euo pipefail

workspace_root="/home/sunyuxiang/rl_alpha"
code_root="${workspace_root}/ours"
python_bin="${workspace_root}/.venvs/quantevolver/bin/python"
config="${code_root}/configs/experiment/recent_alpha_quantevolver_rolling.yaml"
experiment_id="${QE_EXPERIMENT_ID:-recent_alpha_quantevolver_v1}"

export PYTHONPATH="${code_root}/src:${PYTHONPATH:-}"
cd "${code_root}"

"${python_bin}" -m rlalpha.cli matrix run --config "${config}" --experiment-id "${experiment_id}"
"${python_bin}" -m rlalpha.cli evaluate run --config "${config}" --experiment-id "${experiment_id}"
"${python_bin}" -m rlalpha.cli report build --config "${config}" --experiment-id "${experiment_id}"
