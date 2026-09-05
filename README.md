# RLAlpha

当前改进方案与运行入口：[三折滚动 reward / 精简 prompt（v8）](docs/rolling_oof.md)。主方案为 `r1_oof`，配对 LCB 对照为 `r2_paired_oof`；外层保持全历史拟合。

RLAlpha is a leakage-audited S&P 500 factor-search research implementation
based on `../REMOTE_CODEX_BUILD_GUIDE.md` and the stricter repair contract in
`../repair.md`. AlphaGen and QuantEvolver are read-only reference checkouts.

The data/DSL/risk/reward/evaluation paths have been rebuilt and validated on
real data. Random and GP pass a two-cell end-to-end CPU smoke. Formal
`grpo_llm` now uses one persistent online-dataset session around
QuantEvolver/Verl's real PPO trainer; the old custom LoRA policy gradient has
been deleted. Base-LLM and formal GRPO now share one compact
prompt and one fairness protocol: one frozen-pool round produces eight answers,
scores all eight, and admits at most one factor. The project is still **not
ready for a formal full experiment** because an exact real-GPU interrupted-vs-
uninterrupted comparison, the complete 12-cell small matrix, and clean-clone
reproduction remain open.

## Current acceptance state

- Raw six-table audit, v3 panel, v3 Balanced-22 risk panel, deterministic
  manual trace: passed.
- Membership-aware NumPy/Torch DSL, NaN semantics, cache identity, point-in-
  time fundamentals/CCM, leakage sentinels: passed CPU tests.
- Fixed trade/metric universes, zero-fill fixed-weight combination, PSD-Gram ridge,
  serialized label-free deployment transforms, joint RNIC projection,
  per-factor HAC/bootstrap/BH-FDR, QP post-solve gates, lineage and formal
  report refusal rules: passed unit tests and Random/GP real-data smoke.
- GP dispatch uses AlphaGen's repository-local modified `gplearn` engine. One
  complete generation contains eight unique typed candidates; their fitness is
  the shared add-only common-support delta, followed by at most one admission. The old
  custom 128-individual/rescore implementation is removed.
- Qwen3.5-2B file hashes: verified. The only maintained prompt is
  `unified_compact_v1`; there are no hint- or reward-specific prompt variants.
- Formal Verl GRPO: implemented; focused tests exercise old-policy
  ratio/clipping, reference KL, entropy logging, LoRA optimizer/scheduler and
  checkpoint resume. Each update learns from exactly eight completions of one
  prompt and performs at most one pool admission. A fresh GPU smoke and the
  full 12-cell matrix have not run under this new protocol.

The authoritative gap list is
[`docs/revision_compliance_matrix.md`](docs/revision_compliance_matrix.md).
The GP call boundary is documented in
[`docs/gp_integration.md`](docs/gp_integration.md).

## Setup and CPU acceptance

```bash
conda activate rlalpha
cd /home/sunyuxiang/rl_alpha/ours
python -m pip install -e .

python -m rlalpha.cli doctor \
  --config configs/experiment/revision_v3_cpu_smoke.yaml
python -m rlalpha.cli data audit \
  --config configs/experiment/revision_v3_cpu_smoke.yaml \
  --output /data/sunyuxiang/rl_alpha/runs/data_audit
pytest -q
```

Build a new panel; never point these commands at a legacy result and relabel
it:

```bash
python -m rlalpha.cli data build \
  --config configs/experiment/revision_v3_cpu_smoke.yaml
python -m rlalpha.cli risk build \
  --config configs/experiment/revision_v3_cpu_smoke.yaml
python scripts/audit_panel_trace.py \
  --raw-root /data/sunyuxiang/rl_alpha \
  --processed-root /data/sunyuxiang/rl_alpha/processed \
  --output /data/sunyuxiang/rl_alpha/runs/data_audit/panel_trace_sample.parquet
```

The accepted local artifacts have panel fingerprint
`180957ab...d40456` and risk fingerprint `019f26ba...0dc21`; full hashes and
exceptions are in [`docs/data_audit_report.md`](docs/data_audit_report.md).

## Verified CPU search/evaluation smoke

```bash
python -m rlalpha.cli matrix run \
  --config configs/experiment/revision_v3_cpu_smoke.yaml \
  --experiment-id revision_v3_cpu_smoke_new
python -m rlalpha.cli evaluate run \
  --config configs/experiment/revision_v3_cpu_smoke.yaml \
  --experiment-id revision_v3_cpu_smoke_new
python -m rlalpha.cli report build \
  --config configs/experiment/revision_v3_cpu_smoke.yaml \
  --experiment-id revision_v3_cpu_smoke_new
```

Every rerun needs a new experiment ID unless every effective config and
data/code/model/prompt/reward/evaluator fingerprint matches. Method-specific
test opening checks only the requested method cells. The accepted diagnostic run
`revision_v3_cpu_smoke_final` completed Random/GP R0 at budget 8. It predates the
current missing-return policy and should not be used as portfolio-performance
evidence without rerunning evaluation.

## Timing and statistical contract

A signal uses information through close `t`, executes at close `t+1`, and
starts earning `DlyRet` at `t+2`. The label compounds `t+2` through `t+21` and
never drives portfolio PnL. Cross-sectional operators see only that day's
member-and-eligible universe while rolling operators retain earlier available
history. Fundamentals have a six-month lag and 18-month expiry, and CCM links
must still be active on the stock date.

Final primary factor metrics are neutralized Pearson/rank RNIC. Raw IC is
diagnostic. Every final factor receives daily RNIC, HAC inference, moving-block
bootstrap and pool-wise BH-FDR. Dollar-neutral and Balanced-22 fully-neutral
portfolios are reported separately. Missing held returns use a zero daily
contribution at the position's last observable value without reweighting other
holdings; missing name/weight coverage remains explicit in the artifacts.
Historical borrow availability remains an explicit limitation. See
[`docs/evaluation_protocol.md`](docs/evaluation_protocol.md).

## GRPO and expensive jobs

`scripts/smoke_grpo.py` is a real two-update QuantEvolver/Verl smoke, not a
mock or forward-only check. On this host it must be launched with an explicit
GPU mapping, for example:

```bash
CUDA_VISIBLE_DEVICES=<gpu-index> RLALPHA_PHYSICAL_GPU=<gpu-index> \
  python scripts/smoke_grpo.py \
  --run-dir /data/sunyuxiang/rl_alpha/runs/grpo_persistent_smoke_new_id \
  --updates 2
```

`configs/experiment/revision_v3_full_smoke.yaml` declares the future four
methods × three rewards matrix but keeps `auto_start_expensive_jobs: false`.
Do not enable the full matrix until the remaining gates in
[`docs/revision_compliance_matrix.md`](docs/revision_compliance_matrix.md) are
closed. The exact GRPO call boundary and evidence are documented in
[`docs/grpo_integration.md`](docs/grpo_integration.md).

Qwen is pinned to `Qwen/Qwen3.5-2B` revision
`15852e8c16360a2fea060d615a32b45270f8a8fc` at
`/data/shared/huggingface/Qwen3.5-2B`; config, tokenizer, and weight SHA-256
values are verified at runtime. GPUStack services must not be stopped or
modified.

Migration/trust rules are in
[`docs/migration_and_cleanup.md`](docs/migration_and_cleanup.md). Existing
`preliminary_screen`/`confirmatory` output is legacy and is not a continuation
of revision v3.
