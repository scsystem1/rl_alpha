# Experiment protocol

The current rolling OOF comparison crosses `random`, `base_llm`, `grpo_llm`
with `r1`, `r1_oof`, `r2_paired_oof`, seeds 0/1/2. The legacy `gp`, `r0` and
`r2_lcb` implementations remain available. See [rolling OOF protocol](rolling_oof.md)
for fixed folds, paired statistics, outer weight fitting and pending experiments.
Every proposal group has eight
candidates, is scored against one frozen train-only pool, and admits at most
one candidate. Pool capacity is 20. Every method runs the same fixed number of
search steps (250 by default), so each formal cell receives exactly 2,000 raw
candidate proposals. Valid unique market-evaluated formulas remain an outcome
metric and no longer control stopping.

GP uses AlphaGen's modified `gplearn` implementation. Its entire population is
one eight-candidate generation. AlphaGen performs tournament selection,
crossover, subtree/hoist/point mutation and reproduction; the shared RLAlpha
coordinator supplies the add-only common-support candidate delta as fitness. Invalid
DSL trees and already generated expressions are rejected before the formal
proposal group, so every recorded GP round contains eight new typed formulas.

Base-LLM and GRPO use the identical `unified_rolling_summary_v8` prompt. Base-LLM
samples that one prompt eight times. GRPO uses one prompt row with eight
rollouts, learns from those eight rewards in one optimizer update, and then
admits at most one of the same eight candidates. There are no per-answer hint
variants.

This protocol is registered but not fully accepted. Random/GP CPU execution is
available, Base-LLM generation is implemented, and formal GRPO dispatch now
uses the real QuantEvolver/Verl persistent online driver with a two-update CUDA smoke.
Exact real-GPU interrupted equivalence and the complete small matrix remain
open. Therefore the old preliminary/confirmatory commands
must not be used to claim M0–M9 completion.

## Split use

- Train: factor evaluation, validity/duplicate checks, reward, candidate
  delta, pool admission, and GRPO reward if/when enabled.
- Validation: selection among already frozen pool snapshots using the
  predeclared objective; both new OOF rewards use mean RNIC with full-train
  weights. It never updates model or pool.
- Test: may be opened independently for any requested method after that
  method's configured cells complete the configured steps and identities.

Each cell uses a factor-independent train+validation metric universe and a
label-free test trade universe. Factor residual nulls are zero opinions for
ridge/RNIC, so a sparse final expression cannot shrink the metric sample.
Valid metric and executable portfolio counts are reported separately. Detailed
transforms, statistics, portfolio constraints, and missing-return behavior are
in `docs/evaluation_protocol.md`.

## Revision-v3 acceptance runs

`revision_v3_cpu_smoke.yaml` contains Random-R0 and GP-R0, seed 0, for one
eight-candidate step. It exists to exercise real data, lineage, final evaluation, and formal report
gates cheaply. The accepted run `revision_v3_cpu_smoke_final` completed both
cells; this is engineering evidence, not a method comparison with useful
power.

`revision_v3_full_smoke.yaml` declares 12 cells (four methods × three rewards),
seed 0, for eight steps. Expensive jobs are disabled. It may run only after the
remaining prompt/resume gates in the compliance matrix are closed. A later confirmatory
configuration must use predeclared cells/seeds/steps and a new experiment
family; it must not resume the legacy screen.

## Identity, resume, and reporting

Each cell identity covers the experiment/search/reward/data/evaluation/model
configs, panel/risk/index strong hashes, repository commits and dirty patch
hashes, and actual model files for LLM methods. Run identity additionally
freezes the effective config and evaluator semantics. A checkpoint is readable
only with matching checkpoint schema 8, rolling-paired reward semantics, reward/prompt contracts, and
checkpoint fingerprint. Matrix
completion requires accepted train metrics, final pool, manifest, exact step count
and current identity.

Method-specific reports require only that method's requested cells and test
markers. A combined report still requires all methods selected for that report
and identical panel/evaluator identities.
