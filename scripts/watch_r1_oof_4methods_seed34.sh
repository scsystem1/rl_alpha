#!/usr/bin/env bash
set -uo pipefail

primary_session="${1:-r1_oof_s34}"
experiment_id="${2:-r1_oof_4methods_seed34_20260907}"
config="${3:-configs/experiment/r1_oof_4methods_seed34.yaml}"
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
run_root="/data/sunyuxiang/rl_alpha/runs/${experiment_id}"
status_file="${run_root}/evaluation_status.json"
watch_log="${run_root}/recovery_watch.log"

mkdir -p "${run_root}"
exec >>"${watch_log}" 2>&1
echo "$(date -Iseconds) watching tmux session ${primary_session}"
while tmux has-session -t "${primary_session}" 2>/dev/null; do
  sleep 60
done
if [[ -f "${status_file}" ]] && jq -e '.status == "complete"' "${status_file}" >/dev/null 2>&1; then
  echo "$(date -Iseconds) primary launcher completed evaluation; no recovery needed"
  exit 0
fi

# The primary launcher waits for every worker before exiting.  Therefore GPU 3
# is free of this experiment here and is the safe fallback for a failed GRPO 3.
echo "$(date -Iseconds) primary launcher ended without complete evaluation; resuming incomplete cells with GRPO 3 on GPU 3"
cd "${code_root}"
RLALPHA_GRPO3_GPU=3 exec bash scripts/run_r1_oof_4methods_seed34.sh \
  "${experiment_id}" "${config}" r1_oof
