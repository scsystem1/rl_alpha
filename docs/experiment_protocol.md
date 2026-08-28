# Experiment protocol

The formal design crosses search method (`random`, `gp`, `base_llm`,
`grpo_llm`) with reward (`r0`, `r1`, `r2_lcb`). Every proposal group has eight
candidates, is scored against one frozen train-only pool, and admits at most
one candidate. Pool capacity is 20 and budget means valid unique
market-evaluated formulas, not raw generations.

GP uses AlphaGen's modified `gplearn` implementation. Its entire population is
one eight-candidate generation. AlphaGen performs tournament selection,
crossover, subtree/hoist/point mutation and reproduction; the shared RLAlpha
coordinator supplies the add-only common-support candidate delta as fitness. Invalid
DSL trees and already generated expressions are rejected before the formal
proposal group, so every recorded GP round contains eight new typed formulas.

Base-LLM and GRPO use the identical `unified_compact_v1` prompt. Base-LLM
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
  predeclared reward objective. It never updates model or pool.
- Test: may be opened independently for any requested method after that
  method's configured cells complete the correct budget and identities.

Each cell uses its own final 20-factor complete-case train+validation fit
support and label-free test trade support. Valid day/observation counts are
reported for interpretation and do not block other methods. Detailed transforms,
statistics, portfolio constraints, and missing-return behavior are in
`docs/evaluation_protocol.md`.

## Revision-v3 acceptance runs

`revision_v3_cpu_smoke.yaml` contains Random-R0 and GP-R0, seed 0, budget 8. It
exists to exercise real data, lineage, final evaluation, and formal report
gates cheaply. The accepted run `revision_v3_cpu_smoke_final` completed both
cells; this is engineering evidence, not a method comparison with useful
power.

`revision_v3_full_smoke.yaml` declares 12 cells (four methods × three rewards),
seed 0, budget 64. Expensive jobs are disabled. It may run only after the
remaining prompt/resume gates in the compliance matrix are closed. A later confirmatory
configuration must use predeclared cells/seeds/budgets and a new experiment
family; it must not resume the legacy screen.

## Identity, resume, and reporting

Each cell identity covers the experiment/search/reward/data/evaluation/model
configs, panel/risk/index strong hashes, repository commits and dirty patch
hashes, and actual model files for LLM methods. Run identity additionally
freezes the effective config and evaluator semantics. A checkpoint is readable
only with a matching v3 semantics commit and checkpoint fingerprint. Matrix
completion requires accepted train metrics, final pool, manifest, exact budget
and current identity.

Method-specific reports require only that method's requested cells and test
markers. A combined report still requires all methods selected for that report
and identical panel/evaluator identities.
