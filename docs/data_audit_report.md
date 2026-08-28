# Data audit report

This report records the audit actually run on `/data/sunyuxiang/rl_alpha`; it
is not a description of intended checks. The raw hard gates and the full
versioned panel/risk builds passed on 2026-08-10.

## Raw inventory and keys

Inventory hash: `496691ca1974b13f7c2bde9384fc48d7df65d7fbe2cca63e84fb0d8873d3702e`
Schema hash: `8e3cc8f6b16ab08aefa8c167c05dd6c8e68cc8bbe4982172762a7ad18af3d718`

| dataset | rows | date coverage | duplicate primary keys |
|---|---:|---|---:|
| daily | 3,131,836 | 2008-01-02 to 2025-12-31 | 0 |
| membership | 911 | intervals through 2025-12-31 | 0 |
| market | 4,529 | 2008-01-02 to 2025-12-31 | 0 |
| CCM | 907 | links through 2025-12-31 | 0 |
| fundamentals | 14,531 | 2006-01-31 to 2025-12-31 | 0 |
| delistings | 330 | 2008-03-10 to 2025-12-08 | 0 |

All required date columns are physical timestamps. Daily `PERMNO/PERMCO`,
membership/delisting `PERMNO`, CCM `lpermno`, and CCM/fundamental `gvkey`
passed integral/digit-preservation checks. There are no conflicting
`gvkey-datadate` records, duplicate CCM source keys, invalid dates, overlapping
active membership intervals, duplicate market dates, or within-security date
reversals. Machine-readable evidence is in
`/data/sunyuxiang/rl_alpha/runs/data_audit/{raw_file_inventory.json,schema_report.json,key_and_missingness_report.parquet,date_coverage_report.parquet}`.

## Prices, returns, adjustments, and delistings

Raw finite coverage is 99.8395% for close, 99.8426% for daily total return,
and 99.8398% for volume. There are no finite prices at or below zero, negative
finite volumes, returns below -100%, or non-null infinities. The median
`DlyCap / (abs(DlyClose) * ShrOut)` is exactly 1.0 in the source units.

There are 259 invalid/missing price adjustment factors and 259 invalid/missing
share factors (0.0083% of rows, below the predeclared 0.1% hard threshold).
They remain missing and are ineligible; they are never replaced by zero. Of
3,126,808 complete adjusted OHLC rows, four violate the high relation and one
violates the low relation, a combined rate of `7.9954e-7`, below the explicit
`1e-5` anomaly threshold. These five rows remain visible in QA rather than
being silently repaired.

`DlyRet` is treated as the final CIZ total return when finite. A delisting
return is used only for a row explicitly flagged as delisted whose `DlyRet` is
missing and whose same-`PERMNO`/same-date `DelRet` is finite; a finite CIZ
return is never compounded a second time. Two flagged missing rows occur,
both in test, but neither has a matched finite `DelRet`, so both remain missing.
Consequently filled rows are 0 in every split and the no-fill sensitivity is
identical. Final return quantiles at 0.1%, 1%, 50%, 99%, and 99.9% are
`-0.15786`, `-0.07001`, `0.00061`, `0.07306`, and `0.17342`.

## Membership, eligibility, labels, and timing

Membership is interval-inclusive and its checkpoint counts are 500
(2010-06-30), 505 (2018-06-29), 505 (2021-06-30), and 503 (2025-06-30).
Eligibility additionally requires `EQTY/COM/NS`, exchange A/N/Q, active
trading status, positive adjusted close and volume, and a finite current total
return. The full panel has median 503 members and 498 eligible members. From
2010 onward there are zero days outside the 400–600 member hard range and zero
days below a 90% eligible/member ratio. Daily counts are preserved in
`/data/sunyuxiang/rl_alpha/processed/panel/universe_counts.parquet`.

A signal observed after close `t` is executed at close `t+1`; its 20-trading-
day label compounds returns `t+2` through `t+21`. Labels crossing a split end
are missing. The deterministic 12-row trace recomputed labels directly from
the stored daily returns; maximum absolute stored/manual difference was
`3.2732e-9`.

## Fundamentals and Balanced-22

Annual fundamentals use INDL/STD/D/C/USD records, a fixed six-month
availability lag, and an 18-month maximum age. CCM links require `USEDFLAG=1`,
type LC/LU/LS, primacy P/C, deterministic tie-breaking, and are checked both
at report date and at the stock date to prevent an exposure outliving its
link. Book equity must be positive for profitability; investment requires
positive current and prior assets; leverage requires positive assets.
`/data/sunyuxiang/rl_alpha/processed/panel/fundamental_lineage.parquet` retains the report,
availability, link interval, and derived accounting values.

The risk panel uses the audited final return and adjusted close/volume layer.
Rolling beta and residual volatility come from one common-observation OLS
window with intercept (252 observations, minimum 126); residual standard
deviation uses that same window. On all formal dates there were zero
Balanced-22 rank failures, zero condition-number failures above `1e8`, and
100% complete exposures for member rows (hard minimum 99.9%).

## Committed artifacts and manual trace

- Panel artifact version 3, fingerprint
  `180957ab2d7f8046621299311e3f75753e70beed734d044b25b004ad28d40456`.
- Risk artifact version 3, fingerprint
  `019f26bae0875b2727e68ce0413a9dd42c7b44d2f62694423d000a1dea50dc21`.
- Shape: 4,529 dates × 891 securities; risk shape adds 22 exposures.
- Trace: `/data/sunyuxiang/rl_alpha/runs/data_audit/panel_trace_sample.parquet`, metadata fingerprint
  `97ca9b8477281e392747e9a31d83d5996964876730df4d6eca4e45951c4ea59a`.

The trace chooses the three lowest eligible member PERMNOs on four fixed dates
and joins raw rows, adjusted dense fields, membership interval, manual/stored
label, all 22 exposures, and available Compustat/CCM provenance. It is
reproducible with `scripts/audit_panel_trace.py`.

## Remaining data assumptions

Historical borrow availability is unavailable; short-side evaluation must
label the assumption rather than infer borrowability. The five OHLC relation
anomalies and two unresolved delisting rows are retained as missing/audited
exceptions. This data acceptance does not authorize a formal experiment by
itself: run config, evaluator, reward, model, prompt, trainer, checkpoint, and
matrix identities must also match.
