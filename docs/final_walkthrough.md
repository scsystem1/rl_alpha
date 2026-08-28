# Revision-v3 walkthrough

This walkthrough follows the maintained execution path and links every claim
to code, a semantic test, or an artifact. The older
`docs/code_walkthrough_zh.md` is preserved as historical audit input and does
not override this revision status.

## 1. Identity before computation

`config.py` recursively composes YAML and validates every supported section
with Pydantic `extra=forbid`. `matrix/runner.py` derives each expected cell
identity from all referenced configs, panel/risk/index strong hashes,
repository commit/dirty-patch identities, budget, seed, and actual LLM files.
`search/run.py` writes the resolved/effective config and a run identity before
loading a checkpoint. `manifest.py` records raw files, panel artifacts,
repository state, packages, model/tokenizer/weights, prompt, reward, and
evaluator versions.

Tests: unknown config rejection, all repository YAMLs, tampered checkpoint,
matrix retry/skip identity, and report missing-cell rejection in
`tests/unit/test_search.py` and `tests/integration/test_reporting.py`.

## 2. Raw data to committed panel

`data/audit.py` discovers six tables and writes inventory, schema,
missingness/key, and date-coverage artifacts. `data/validate.py` hard-checks
keys, identifier representation, timestamps, within-security order,
membership intervals, illegal numeric values, adjustment anomaly thresholds,
coverage and S&P membership counts.

`data/panel.py` validates first, reads source files, applies audited CRSP
adjustments/delisting rules, sorts explicitly, and builds all dense arrays in
one fresh temporary directory. `membership.zarr` and `eligibility.zarr` remain
separate contracts. Labels compound `t+2:t+21` and are cleared at split exits.
A source/code/semantics mismatch never reuses old daily data; the entire panel
is committed by rename and the previous artifact is marked legacy.

Evidence: `docs/data_audit_report.md`, panel/risk manifests under
`/data/sunyuxiang/rl_alpha/processed/panel`, daily `universe_counts.parquet`, and the
12-row `/data/sunyuxiang/rl_alpha/runs/data_audit/panel_trace_sample.parquet` produced by
`scripts/audit_panel_trace.py`.

## 3. Point-in-time risk

`data/fundamentals.py` filters the Compustat convention, rejects conflicting
duplicates, sets six-month availability, computes denominators conservatively,
and deterministically resolves CCM candidates. `risk/builder.py` additionally
enforces the link on the stock date and expires accounting records after 18
months. Its market styles use the committed final return and adjusted
price/volume layer. `_rolling_market_regression` computes beta and residual
standard deviation from one identical window.

Each active date is winsorized/imputed/z-scored into the fixed intercept + 11
industry dummy + 10 style Balanced-22 matrix. Formal dates hard-fail rank below
22, condition number above `1e8`, or member completeness below 99.9%.
`fundamental_lineage.parquet` preserves the report and CCM source.

Tests: hand OLS residual volatility, stock-date CCM expiry, denominator rules,
FF12 and orthogonality in `tests/unit/test_risk.py` and
`tests/unit/test_data_contracts.py`.

## 4. Typed DSL and train-only search

The parser produces a canonical typed AST. `dsl/evaluator.py` and
`torch_evaluator.py` keep time-series history available but apply the daily
member/eligibility mask at every CSRank/CSZScore node. A comparison returns
NaN unless both operands are finite. Cache keys include evaluator semantics,
panel namespace, and mask fingerprint.

`SearchCoordinator` owns parsing/evaluation/validity/budget/reward/admission
for every proposer. It records stable proposal/group IDs and the frozen pre-
group pool hash. `dsl/validity.py` uses train-only coverage and multiple
absolute correlation diagnostics. `PoolManager` recomputes the complete pool
objective, converts non-finite outcomes to explicit penalties/reasons, and
admits at most one candidate per frozen group while preserving pre/post pool
hashes.

Random and typed-AST GP are maintained. Base-LLM uses the same grammar and a
versioned prompt. `grpo_llm` currently blocks before search rather than
dispatching the isolated legacy custom policy gradient.

Evidence: DSL/property tests, leakage sentinel, reward/pool tests, lineage
referential-integrity test, and real Random/GP smoke.

## 5. Freeze, fit, and open test once

`evaluation/finalize.py` first resolves the exact configured matrix and checks
cell state, budget, current identity, required artifacts, manifests, and
panel/evaluator comparability. Only then does it load test. It freezes a common
train+validation mask and a label-free test trade mask for that cell's
final expression and writes their hashes/counts.

`FactorTransformPipeline` applies daily 1/99 winsorization, z-scoring, joint
signal/label Balanced-22 projection on one complete sample, and post-residual
z-scoring. It removes days that become invalid after standardization and is
serialized with the ridge weights. Test uses the identical pipeline without
refitting global choices.

Every factor gets daily raw diagnostic IC, primary Pearson/rank RNIC,
projection diagnostics, HAC inference, fixed moving-block bootstrap, and
pool-wise BH-FDR. IDs join back to final-pool lineage. Details and schemas are
in `docs/evaluation_protocol.md`.

## 6. Portfolio and report failure semantics

`evaluation/portfolio.py` implements four sleeves, next-close execution, and
PnL beginning the following day. Dollar-neutral and fully-neutral portfolios
are distinct. Every QP result is recomputed against net, gross, exposure,
maximum-weight and tradability tolerances; violations reject the rebalance.
Missing held returns make the day and full performance path invalid, and MDD
begins at wealth 1. `reporting/build.py` refuses to rescue missing portfolio
paths by dropping dates when computing paired Sharpe.

`reporting/build.py` produces a formal report only for the exact expected
matrix with complete test markers and compatible identities. Same-date RNIC
comparisons and across-seed results preserve missingness.

Evidence: QP/timing/missing/MDD tests and
`/data/sunyuxiang/rl_alpha/runs/revision_v3_cpu_smoke_final`.

## 7. GRPO boundary and launch decision

`grpo/verl_config.py` composes onto QuantEvolver/Verl's real PPO config and
asserts GRPO advantage, active clipping, active reference KL and two PPO
epochs. `stage_coordinator.py` keeps one trainer alive while `online_dataset.py`
generates the next frozen train-only prompt after each optimizer update,
`verl_reward_function.py` scores the complete rollout barrier, and
`verl_trainer.py` constructs the real `RayPPOTrainer` actor/rollout/reference
workers. Historical CUDA smoke outputs were cleared because they used the old
restart-per-stage semantics. The next acceptance run must verify changing
actor checkpoints, active PPO clip/reference KL, and checkpoint resume using
exactly eight completions per update.

The correct project decision is therefore: data/DSL/evaluation and formal
GRPO optimizer integration are implemented; full experiment launch is denied
until a fresh one-prompt/eight-answer GPU smoke, exact real-GPU interruption
equivalence, the 12-cell matrix and clean-clone reproduction finish. The
remaining evidence is
enumerated in `docs/grpo_integration.md`, `docs/prompt_benchmark.md`, and
`docs/revision_compliance_matrix.md`.
