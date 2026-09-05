# Revision compliance matrix

This matrix audits `../REMOTE_CODEX_BUILD_GUIDE.md` and `../repair.md` against
executable code and produced evidence. Status meanings are: **verified** (the
required semantic test/artifact was run), **implemented—not fully exercised**
(code and focused tests exist but the required full acceptance run does not),
and **blocked** (a Definition-of-Done condition is intentionally unavailable).

| Requirement | Implementation | Status | Evidence / impact | Remaining action | Acceptance |
|---|---|---|---|---|---|
| Strict environment/config and read-only doctor | `config.py`, `doctor.py` | verified | Every repository YAML is parsed by `extra=forbid`; unknown fields fail | Clean-clone install still needs independent reproduction | config tests + CPU suite |
| Six raw tables, schemas, keys, dates, IDs, missingness | `data/audit.py`, `data/validate.py` | verified on full data | 3,131,836 daily rows; all six primary keys unique; timestamp/ID/order gates pass | None for current bundle | four machine reports + `data_audit_report.md` |
| CRSP adjustment/final return/delisting policy | `data/adjustments.py`, panel manifest | verified on full data | finite CIZ return never double-counted; 2 unresolved test rows, 0 fills; adjusted anomalies/counts retained | A different missing-delisting policy would require a new family | golden tests + v3 manifest |
| Membership and explicit trade eligibility drive CS ops | `data/eligibility.py`, `dsl/evaluator.py`, Torch evaluator | verified | nonmember extremes/future values cannot alter member rank/z-score; rolling pre-membership history retained | None | NumPy/Torch mask/tie/NaN tests |
| Comparison NaN semantics and evaluator/cache identity | evaluators, `SignalCache`, panel namespaces | verified | non-finite operand stays NaN; keys include semantics, panel and mask fingerprint; no float32 resume downcast | None | nested comparison/cache tests |
| Signal/label timing and end-to-end leakage | panel labels, `tests/leakage/test_sentinels.py` | verified | mutating test raw rows leaves train signal, reward, admission and update-input hashes unchanged | None | loader→panel→evaluator→pool sentinel |
| Point-in-time fundamentals and CCM | `data/fundamentals.py`, `risk/builder.py` | verified | deterministic link selection, six-month lag, 18-month expiry, positive denominators; exposure cannot outlive stock-date link | Expand trace dates only if audit policy changes | golden test + `fundamental_lineage.parquet` |
| Balanced-22 definitions and same-window residual volatility | `risk/exposures.py`, `risk/builder.py` | verified on full data | 0 formal rank failures, 0 condition failures, 100% complete member exposure; adjusted price/volume and final return used | None for current panel | hand OLS test + v3 risk build |
| Panel build is content-identified and transactional | `data/panel.py`, risk builder, `PanelStore` | verified | mismatch performs full temp rebuild and preserves legacy; readers require committed panel/risk versions and matching fingerprints | External legacy panel must be rebuilt, not relabeled | source-mutation test + real v3 build |
| Pool reward uses a fixed metric universe and finite failure semantics | `rewards/base.py`, `factors/moments.py`, `factors/pool.py` | verified | missing factor residuals are zero opinions; day-equal PSD Gram and cross moments; NaN/Inf gets penalty and reason | None | hand Gram/recomputation/zero-variance tests |
| Near-duplicate rule avoids sign cancellation | `dsl/validity.py` | verified | records mean absolute daily Pearson, pooled Pearson, mean absolute daily rank and coverage | None | alternating-sign and ±factor tests |
| One serialized fit/OOS transform | `factors/transform.py`, `RidgeCombiner` | verified | each factor uses label-free support; residuals receive z-score without a second winsorization; zero-variance days become zero opinions; round trip identical | None | label-missingness/serialization tests + real CPU smoke |
| Fixed-universe evaluation support | `evaluation/finalize.py` | unit verified | factor missingness cannot shrink primary RNIC; metric and executable portfolio observations are reported separately | real multi-method comparison pending | method-filter/fixed-universe tests + smoke artifacts |
| RNIC uses identical signal/label projection sample | transform/finalize | verified | zero-filled composite and label share `trade_mask & finite(label)` and one exposure projector | None | unequal-missingness orthogonality test |
| Fully-neutral QP is post-solve accepted | `evaluation/portfolio.py` | verified | explicit net/exposure/gross/max-weight/tradability checks and solver audits; inaccurate status alone cannot pass | Full matrix feasibility rate remains unknown | tolerance-failure tests + CPU smoke audits |
| Missing held returns and MDD | portfolio/reporting | verified | held missing return makes daily/path metrics invalid; paired Sharpe also refuses any missing path; wealth begins at 1 | A usable formal portfolio requires a preapproved exit policy or complete returns | unit tests + CPU smoke produced NaN rather than zero-imputed metrics |
| Per-factor RNIC, HAC/bootstrap, BH-FDR | `evaluation/statistics.py`, finalize | implemented and CPU-smoke verified | daily Pearson/rank RNIC, fixed HAC/block/bootstrap/seed, pool-wise per-metric FDR, frozen direction | Full 12-cell output schema not yet exercised | statistic tests + smoke Parquet/CSV |
| Proposal/admission/snapshot/final lineage | coordinator, pool, `search/run.py` | implemented and CPU/GPU-smoke verified | stable proposal/factor/event/snapshot/final IDs, explicit GRPO group/stage/update fields, pre/post hashes and linked artifacts | Full-matrix lineage join remains unexercised | referential-integrity tests + GRPO smoke lineage |
| Manifest, resume, matrix and aggregate identity | `manifest.py`, coordinators, matrix/reporting | implemented—not fully exercised | strong panel/model hashes; repository dirty identity; prompt/reward/evaluator versions; atomic outer/Verl checkpoint hashes; wrong/missing cells rejected | Run exact real-GPU interrupted-vs-uninterrupted comparison and clean-clone reproduction | tamper/failure/report tests + real checkpoint reload |
| Unified compact 2B prompt | `search/prompts.py`, `benchmark_prompts.py` | tokenizer verified; fresh GPU evidence pending | one `unified_rolling_summary_v8` contract shared by Base-LLM/GRPO; full pool, horizon/risk definition and canonical OOF summary; no theme hints | Run fresh GPU smoke and matched summary ablation | prompt tests + `rolling_oof_token_profile.json` |
| Rolling OOF mean / paired LCB reward | `rewards/walk_forward.py`, shared factory, pool comparison | CPU verified; market comparison pending | three purged folds, paired gap-aware HAC, fixed support, independent-reference and worker replay checks; outer weights remain full history | Run `rolling_oof.yaml` with new experiment ID | [protocol and limitations](rolling_oof.md) |
| QuantEvolver/Verl owns complete GRPO loss | `grpo/verl_config.py`, reward hook, trainer wrapper, persistent coordinator | focused loss tests pass; real smoke reaches FSDP/vLLM startup but GPU-memory contention blocked updates | real `RayPPOTrainer` owns old ratio, clip, reference KL, entropy and optimizer state | V0 trainer is deprecated; migrate to V1 when `transfer_queue` is available | rerun the two-update smoke on an actually idle GPU and retain its report |
| Atomic GRPO stage/checkpoint/resume | `grpo/stage_coordinator.py` | implemented—not fully exercised | explicit events; no-admission stage closes; deterministic attempts; checkpoint directory hash is verified before outer commit; second real GPU stage loads step one | Run separate interrupted/uninterrupted GPU jobs and compare generated sequences/admission/lineage exactly | CPU equivalence/tamper tests + real step-1→step-2 resume |
| Full 4×3 small matrix, Base-LLM and GRPO GPU smoke | `revision_v3_full_smoke.yaml`, `smoke_grpo.py` | new-protocol smoke and matrix not run | each maintained LLM round is one identical prompt with eight answers and at most one admission | Run fresh smoke, close real resume equivalence, then run all cells with matching identities/budgets | old artifacts cleared; new Definition-of-Done evidence absent |
| Cleanup and one maintained formal path | `.gitignore`, migration note, persistent Verl dispatch | implemented | legacy controller/bridge/adapter and their tests are deleted; old checkpoints are rejected | clean-clone run still needed | import/test audit + `migration_and_cleanup.md` |

## Artifact trust classification

All pre-revision runs and `/data/sunyuxiang/rl_alpha/processed` are
`legacy_noncomparable`. The accepted local data artifacts are panel v3
`180957ab...d40456` and risk v3 `019f26ba...0dc21`. The two-cell
`revision_v3_cpu_smoke_final` is engineering evidence only; its portfolio paths
are deliberately invalid because held returns are missing.

## Definition-of-Done decision

The project is **not complete under `repair.md`**. Data/DSL/evaluation,
Random/GP CPU execution, and real staged QuantEvolver/Verl optimization are
now evidence-backed. Four mandatory conditions remain open: a fresh GPU smoke
under the one-prompt/eight-answer semantics; exact real-GPU interrupted-vs-
uninterrupted equivalence; the full Base-LLM/GRPO small matrix; and a clean-
clone reproduction. Formal
large-budget experiments must not start.
