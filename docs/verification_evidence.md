# Verification evidence

Commands use `/home/sunyuxiang/miniconda3/envs/rlalpha/bin/python` and the
repository root `/home/sunyuxiang/rl_alpha/ours`.

## CPU and static checks

| Scope | Command | Result |
|---|---|---|
| default CPU suite | `pytest -q` | 92 passed; one real-data marker deselected |
| unit suite | `pytest -q tests/unit` | 82 passed |
| leakage suite | `pytest -q tests/leakage` | 7 passed |
| integration suite | `pytest -q tests/integration` | 3 passed; one real-data marker deselected |
| real-data loader/DSL smoke | `pytest -q -m real_data` | 1 passed |
| import/syntax | `python -m compileall -q src tests scripts` | passed |
| shell guard syntax | `bash -n scripts/run_full_experiment.sh` | passed |
| patch whitespace | `git diff --check` | passed |

The suite includes golden adjustment/label/CCM/market-model tests, NumPy/Torch
DSL parity, cache and resume identity, end-to-end leakage mutation,
fixed-universe PSD-Gram reward, serialized label-free transforms, zero-opinion handling, QP hard
acceptance, missing-return/MDD behavior, HAC/BH, lineage referential integrity,
matrix failure/recovery, and incomplete-report refusal.

## Full data artifacts

- `scripts/validate_data.py --root /data/sunyuxiang/rl_alpha`: all hard gates
  passed.
- `python -m rlalpha.cli data audit ...`: all four requested machine reports
  and the human audit were generated.
- Full panel build: 3,131,836 rows, shape 4,529 × 891, panel v3 fingerprint
  `180957ab2d7f8046621299311e3f75753e70beed734d044b25b004ad28d40456`.
- Full risk build: shape 4,529 × 891 × 22, risk v3 fingerprint
  `019f26bae0875b2727e68ce0413a9dd42c7b44d2f62694423d000a1dea50dc21`;
  rank/condition failures 0/0 and formal member completeness 1.0.
- Twelve-row raw→panel→label→risk/CCM trace: passed, maximum manual-label
  difference `3.2732e-9`.

## Real-data end-to-end CPU smoke

`revision_v3_cpu_smoke_final` uses the frozen
`configs/experiment/revision_v3_cpu_smoke.yaml`: Random-R0 and GP-R0, seed 0,
budget 8. It must contain two complete cell states, a complete experiment test
transaction, matching manifests, proposal/admission/snapshot/final lineage,
common universe hashes, per-factor RNIC/significance/FDR, portfolio solver
audits, and a formal two-cell report. Its portfolio performance is expected to
be explicitly invalid/NaN when any held return is missing; a finite Sharpe
computed after dropping those dates would be a failure.

## Real GRPO GPU evidence

The persistent-cell smoke now reaches the real FSDP actor and vLLM startup on
the installed Verl version. A 2026-08-27 attempt on physical CUDA 1 stopped
before the first optimizer update because another process consumed the card
after launch, leaving only 3.15 GiB for vLLM. Ray shut down and the failed
60 KiB run directory was removed. Therefore no two-update GPU acceptance is
claimed yet. The smoke itself requires two changing LoRA checkpoints,
optimizer/scheduler/RNG state, finite PPO/KL/entropy metrics, positive gradient
norms, and at most two checkpoints before it can write `status: passed`.

## Deliberately not run

- No actual Base-LLM 500-generation acceptance was rerun after semantic repair.
- No completed persistent-cell two-update GPU smoke or separate-process
  interrupted-versus-uninterrupted exact sequence,
  admission and lineage comparison has run; CPU outer-state equivalence exists.
- No train-only prompt quality benchmark exists beyond tokenizer profiling.
- No full 4×3 method/reward matrix or large-budget experiment was started.

These omissions are blockers, not skips counted as passes. The full-experiment
script exits nonzero until they are closed; the GRPO smoke is now a real
acceptance command.
