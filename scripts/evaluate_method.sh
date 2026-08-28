#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "usage: $0 CONFIG EXPERIMENT_ID METHOD" >&2
  exit 2
fi
config="$1"
experiment_id="$2"
method="$3"

python -u -m rlalpha.cli evaluate run \
  --config "${config}" \
  --experiment-id "${experiment_id}" \
  --method "${method}"
python -u -m rlalpha.cli report build \
  --config "${config}" \
  --experiment-id "${experiment_id}" \
  --method "${method}"
