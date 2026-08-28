#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 3 ]]; then
  echo "usage: $0 CONFIG EXPERIMENT_ID METHOD [extra matrix arguments...]" >&2
  exit 2
fi
config="$1"
experiment_id="$2"
method="$3"
shift 3

exec python -u -m rlalpha.cli matrix run \
  --config "${config}" \
  --experiment-id "${experiment_id}" \
  --method "${method}" \
  "$@"
