#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "usage: $0 CONFIG EXPERIMENT_ID [METHOD ...]" >&2
  exit 2
fi
config="$1"
experiment_id="$2"
shift 2
methods=("$@")
if [[ ${#methods[@]} -eq 0 ]]; then
  methods=(random gp base_llm grpo_llm)
fi

if [[ "${RLALPHA_METHOD_PARALLEL:-0}" == "1" ]]; then
  matrix_args=()
  for method in "${methods[@]}"; do
    matrix_args+=(--method "${method}")
  done
  python -u -m rlalpha.cli matrix run --config "${config}" --experiment-id "${experiment_id}" "${matrix_args[@]}"
else
  for method in "${methods[@]}"; do
    scripts/run_method.sh "${config}" "${experiment_id}" "${method}"
  done
fi

for method in "${methods[@]}"; do
  scripts/evaluate_method.sh "${config}" "${experiment_id}" "${method}"
done
if [[ ${#methods[@]} -gt 1 ]]; then
  report_args=()
  for method in "${methods[@]}"; do
    report_args+=(--method "${method}")
  done
  python -u -m rlalpha.cli evaluate run --config "${config}" --experiment-id "${experiment_id}" "${report_args[@]}"
  python -u -m rlalpha.cli report build --config "${config}" --experiment-id "${experiment_id}" "${report_args[@]}"
fi
