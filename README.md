# RLAlpha

Leakage-safe S&P 500 risk-neutral alpha mining for the experiment in
`../REMOTE_CODEX_BUILD_GUIDE.md`. This repository is independent of the
read-only AlphaGen and QuantEvolver reference checkouts.

## Implemented Contract

The project contains the M0-M9 pipeline: point-in-time panel construction,
canonical typed DSL, Balanced-22 neutralization, R0/R1/R2_LCB rewards, exact
pool admission, resumable Random/typed-AST GP/Base LLM/staged LoRA GRPO,
validation snapshot selection, transactional test finalization, dollar- and
fully-neutral portfolios, matrix state machines, and cross-method reports.

Qwen is pinned to `Qwen/Qwen3.5-2B` revision
`15852e8c16360a2fea060d615a32b45270f8a8fc` at
`/data/shared/huggingface/Qwen3.5-2B`. Critical package versions are recorded
in `requirements-llm.lock`; every run also captures `pip freeze` and GPU state.

## Setup And Acceptance

```bash
conda activate rlalpha
cd /home/sunyuxiang/rl_alpha/ours
python -m pip install -e .
python -m pip install -r requirements-llm.lock

python -m rlalpha.cli doctor --config configs/experiment/preliminary_screen.yaml
pytest
pytest -m real_data

CUDA_VISIBLE_DEVICES=4 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  RLALPHA_VLLM_MEMORY_UTILIZATION=0.18 \
  python scripts/smoke_model.py --n 500 --seed 2026 \
  --output /data/sunyuxiang/rl_alpha/runs/acceptance/model/base_llm_500.json

CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
  python scripts/smoke_grpo.py --updates 2
```

GPUStack services are never stopped or modified. The matrix runner waits for
free-memory thresholds and maps Base LLM to physical GPU 4 and single-GPU GRPO
jobs to physical GPUs 2 and 3. It does not create collectives across the A100
and H800. CUDA OOM retries preserve rollout group eight and reduce only the
within-group microbatch. CPU cells run at most four concurrently with eight
threads each.

## Experiments

```bash
# 12 cells: 4 methods x 3 rewards x seed 0, 5,000 valid unique formulas.
python -m rlalpha.cli matrix run \
  --config configs/experiment/preliminary_screen.yaml \
  --experiment-id preliminary_screen

python -m rlalpha.cli evaluate run \
  --experiment-id preliminary_screen \
  --config configs/experiment/preliminary_screen.yaml
python -m rlalpha.cli report build \
  --experiment-id preliminary_screen \
  --config configs/experiment/preliminary_screen.yaml

# Six core configurations x seeds 0,1,2, budget 20,000. Seed 0 resumes the
# corresponding screening archive and budget rather than starting over.
python -m rlalpha.cli matrix run \
  --config configs/experiment/confirmatory.yaml \
  --experiment-id confirmatory

python -m rlalpha.cli evaluate run \
  --experiment-id confirmatory \
  --config configs/experiment/confirmatory.yaml
python -m rlalpha.cli report build \
  --experiment-id confirmatory \
  --config configs/experiment/confirmatory.yaml
```

Each cell is independent. A failed cell is recorded and does not stop the
others; rerunning the same command resumes its atomic checkpoint. Test data is
opened only by `evaluate run`, after formulas and validation selection are
frozen. Evaluation first fixes one cross-method test universe. A transaction
hash covers all frozen pools, panel/risk manifests and evaluation code, so a
changed input cannot silently reuse prior test output. Reports include raw IC,
RNIC, neutralization retention, paired same-date inference, portfolio costs,
named Balanced-22 exposures, candidate efficiency and GPU/wall time.

## Timing And Leakage

A signal is formed after close `t`, executed at close `t+1`, and starts earning
daily `DlyRet` at `t+2`. The 20-day label compounds `t+2` through `t+21` and is
never used for portfolio PnL. Membership boundaries are inclusive; Compustat
is lagged six months; labels that cross a split end are invalid. Historical
borrow availability is unavailable, so the preliminary test assumes short
borrowability and records that limitation in every final report.
