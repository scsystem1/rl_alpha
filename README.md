# rlalpha

Leakage-safe research code for the S&P 500 risk-neutral alpha-mining experiment specified in `../REMOTE_CODEX_BUILD_GUIDE.md`. The implementation is independent from the read-only AlphaGen and QuantEvolver reference checkouts.

## Current delivery

Milestones M0–M4 are implemented: environment/data doctor, schema-based CRSP/Compustat loading and QA, canonical typed DSL, Balanced-22 construction and neutralization, factor calculator, ridge combiner, exact pool admission/replacement, and R0/R1/R2_LCB objectives. M5–M9 search, LLM/GRPO, portfolios, and matrix execution are intentionally deferred until this foundation is accepted. Expensive experiments never auto-start.

## Setup

```bash
conda activate rlalpha
cd /home/sunyuxiang/rl_alpha/ours
python -m pip install -e .
pytest
```

## Exact commands

```bash
python -m rlalpha.cli doctor --config configs/experiment/preliminary_screen.yaml
python -m rlalpha.cli data validate --config configs/data/sp500.yaml
python -m rlalpha.cli data build --config configs/data/sp500.yaml
python -m rlalpha.cli factor eval --expr 'CSZScore(Div(Delta($close,20),Std($return,20)))' --split train
python scripts/locate_model.py --root /data/shared/huggingface
python scripts/smoke_factor.py
```

`data build` writes only to `/data/sunyuxiang/rl_alpha/processed`; raw Parquet files are never modified. All paths can be overridden with `RLALPHA_*` environment variables documented in `rlalpha.config`.

## Leakage/execution definition

A signal is formed after close t. Entry is at close t+1. The label compounds daily total returns t+2 through t+21 inclusive. Membership boundaries are inclusive, annual Compustat is lagged six calendar months, and split-tail labels are invalidated when the exit crosses the split end.

Known blockers for the next delivery: this host currently cannot communicate with the NVIDIA driver and Qwen3.5-2B is absent from the shared model directory. These do not affect CPU M0–M4 validation.

