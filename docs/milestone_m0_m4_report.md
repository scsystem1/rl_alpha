# M0–M4 acceptance report

> Historical checkpoint only. Its fingerprints and test counts predate the
> revision-v3 semantic repair and are `legacy_noncomparable`; use
> `revision_compliance_matrix.md` and `data_audit_report.md` for current
> acceptance.

## Delivered

- M0: installable Python 3.11 project, configuration, read-only doctor, repository/data/model fingerprints, CLI and reference audit.
- M1: schema discovery, immutable CRSP adjustment, delisting audit, inclusive historical membership, CCM/Compustat point-in-time handling, split-safe next-close labels, partitioned Parquet and dense Zarr panel.
- M2: canonical typed AST, strict parser and `<expr>` extraction, fixed constants/windows, protected operators, deterministic hashing, evaluator, typed random sampler and candidate validity checks. No Python `eval` is used.
- M3: full Balanced-22 builder, FF12 mapping, ten style exposures, daily winsorization/imputation/z-scoring, QR neutralizer with diagnostics, and full risk Zarr.
- M4: daily Pearson IC calculator, explicit-ridge combiner, exact capacity/replacement pool manager, frozen-group admission, R0/R1/R2_LCB, and Newey–West lag-20 standard error.

## Real data acceptance

- Source rows: 3,131,836 daily; 911 membership intervals; 4,529 market days; 907 CCM links; 14,531 annual fundamentals; 330 delistings.
- Duplicate `PERMNO × DlyCalDt`: 0.
- Historical membership counts: 500 (2010-06-30), 505 (2018-06-29), 505 (2021-06-30), 503 (2025-06-30).
- `DlyCap / (abs(DlyClose) * ShrOut)` median: 1.0.
- Daily close/return/volume coverage: all above 99.83%.
- Invalid adjustment factors: 259 price and 259 share-factor rows, emitted as NaN.
- Missing flagged delisting returns: 2; neither matched a finite `DelRet`, both remain explicitly unresolved.
- Dense panel: 4,529 dates × 891 securities; 2,715,491 finite split-safe labels.
- Panel build fingerprint: `dd68112dbd335ec52705a3d869543a0dced65dd35398e1f78468e9795c730a3d`; rerun reused it unchanged.
- Balanced-22 shape: 4,529 × 891 × 22; build fingerprint `07499b36e534bdfef38caee1717dc4f14fa0ad87d07c680842455730f677efe5`; rerun reused it unchanged.
- Five sampled dates had rank 22, condition number 16.7–19.5, and maximum residual exposure between `1.6e-16` and `4.8e-16`.

## Tests

`pytest -q`: 25 passed. Coverage includes unit, leakage sentinel, and synthetic end-to-end tests. The typed grammar generated 10,000 bounded ASTs in the test suite.

## Subsequent milestones

This document records the original M0-M4 checkpoint. Several M5-M9 components
have since been repaired, including formal QuantEvolver/Verl GRPO and a real
CUDA smoke. Prompt quality selection, exact real-GPU resume equivalence, the
full small matrix, and clean-clone acceptance remain open.
Current commands and status live in `README.md`; formal completion must be
judged from the current compliance matrix and matching run artifacts, never
from this historical report.
