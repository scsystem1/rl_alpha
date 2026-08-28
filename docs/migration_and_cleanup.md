# Migration and cleanup

The revisions change experiment semantics, not just implementation details.
Old panels, factor caches, pools, checkpoints, combiners, final metrics, and
reports are not comparable with the revision-v3 family and must not be joined
or resumed.

## Semantic breaks and rerun boundary

| Change | Earliest required rerun |
|---|---|
| membership/eligibility-aware CSRank and CSZScore; NaN comparison semantics | panel/evaluator cache and every factor search |
| final CIZ/delisting policy, adjusted close/volume, label boundary | panel |
| CCM stock-date interval and six-month/18-month accounting policy | risk panel |
| same-window market residual volatility and audited Balanced-22 gates | risk panel |
| joint signal/label RNIC sample and common complete-case pool reward | every search/pool |
| mean-absolute daily duplicate rule and non-finite penalty | every search/pool |
| per-cell complete-case support and serialized transform | validation selection and final evaluation |
| post-solve QP gate, missing-held-return invalidation, initial-wealth MDD | portfolio evaluation |
| unified prompt hash and one-prompt/eight-answer protocol | Base-LLM and GRPO generation |
| AlphaGen gplearn generation and frozen-pool-delta fitness | all GP results and checkpoints |
| real Verl GRPO trainer | all GRPO results; only the new CUDA smoke is revision-v3 evidence |
| manifest, checkpoint, lineage, and matrix identity | complete run |

No prior artifact can be repaired by editing metadata. A mismatch causes a full
temporary rebuild and atomic commit; resume and aggregate paths reject missing
or incompatible identities.

## Trust classification

All historical experiment outputs under repository `artifacts/` and the
`output -> /data/sunyuxiang/rl_alpha/runs` target were intentionally cleared
before adopting the unified prompt and one-prompt/eight-answer semantics. No
old checkpoint is accepted as a continuation seed. Processed source data is a
separate data product and is still validated by its manifest/fingerprint; it
must not be presented as a completed experiment result.

New experiments must use a new ID and either the validated v3 panel or a newly
built panel with the exact same contract. `revision_v3_cpu_smoke.yaml` is the
small Random/GP acceptance configuration. `revision_v3_full_smoke.yaml`
declares the future 12-cell matrix but keeps expensive jobs disabled until the
remaining prompt/resume gates are closed.

## Maintained and isolated paths

The maintained formal search dispatch includes Random, GP, Base-LLM and the
persistent online-dataset `grpo_llm` branch in `run_search`. The per-group
`searcher_for` interface intentionally rejects GRPO because it cannot represent
one cell-wide Verl lifecycle. The legacy controller and bridge/adapter modules
are deleted, with no compatibility import or checkpoint migration retained.
`scripts/smoke_grpo.py` cannot pass without two changing actor checkpoints and
the required loss metrics.

The repository-local processed/runs/cache directories are ignored by Git to
prevent large binary artifacts and self-referential dirty-patch manifests.
Small audit, prompt, and documentation artifacts remain visible. The user's
pre-existing untracked `docs/code_walkthrough_zh.md` was preserved and was not
rewritten as current acceptance evidence.

## Cleanup decisions

- Retained: reference AlphaGen and QuantEvolver checkouts, which are read-only
  evidence and runtime dependencies.
- Replaced as formal behavior: the legacy staged controller, bridge and adapter
  by the persistent online dataset coordinator and real optimizer smoke.
- Superseded: any panel/risk artifact below versions 3/3, expression-only cache
  semantics, and uncommitted checkpoints. Readers reject them.
- Deleted by explicit user request: all old RLAlpha experiment artifacts and
  external run outputs. User-authored source and documentation were retained.

## Formal launch gate

Do not change `auto_start_expensive_jobs` to true until all of the following
remaining items exist: real-GPU interrupted-versus-uninterrupted equivalence;
a train-only prompt quality benchmark; and a full
Random/GP/Base-LLM/GRPO × R0/R1/R2 small matrix with valid portfolio handling.
