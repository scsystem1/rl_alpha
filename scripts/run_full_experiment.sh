#!/usr/bin/env bash
set -euo pipefail

ROOT=/home/sunyuxiang/rl_alpha/ours
RUNS=/home/sunyuxiang/rl_alpha/ours/output
LOG="$RUNS/full_experiment_orchestrator.log"
mkdir -p "$RUNS"
exec > >(tee -a "$LOG") 2>&1
cd "$ROOT"

matrix_complete() {
  local experiment_id=$1
  python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {}; raise SystemExit(0 if d and all(v.get("status") == "complete" for v in d.values()) else 1)' "$RUNS/$experiment_id/matrix_state.json"
}

evaluation_complete() {
  local experiment_id=$1
  python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.exists() else {}; raise SystemExit(0 if d.get("status") == "complete" else 1)' "$RUNS/$experiment_id/test_finalization.json"
}

run_matrix_until_complete() {
  local experiment_id=$1
  local config=$2
  until matrix_complete "$experiment_id"; do
    conda run -n rlalpha python -m rlalpha.cli matrix run --config "$config" --experiment-id "$experiment_id" || true
    matrix_complete "$experiment_id" || sleep 60
  done
}

finalize_until_complete() {
  local experiment_id=$1
  local config=$2
  until evaluation_complete "$experiment_id"; do
    conda run -n rlalpha python -m rlalpha.cli evaluate run --experiment-id "$experiment_id" --config "$config" || true
    evaluation_complete "$experiment_id" || sleep 60
  done
  conda run -n rlalpha python -m rlalpha.cli report build --experiment-id "$experiment_id" --config "$config"
}

run_matrix_until_complete preliminary_screen configs/experiment/preliminary_screen.yaml
finalize_until_complete preliminary_screen configs/experiment/preliminary_screen.yaml
run_matrix_until_complete confirmatory configs/experiment/confirmatory.yaml
finalize_until_complete confirmatory configs/experiment/confirmatory.yaml

python -c 'import json, pathlib, time; pathlib.Path("/home/sunyuxiang/rl_alpha/ours/output/full_experiment_complete.json").write_text(json.dumps({"status":"complete","finished_at":time.time()}, indent=2)+"\n")'
