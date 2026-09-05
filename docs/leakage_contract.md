# Leakage contract

Signals at date t may read data at or before t only. `Ref(x,n)` is a backward lag. The 20-day label is the product of `DlyRet[t+2:t+22]`, corresponding to next-close entry and t+21 close exit. Exit dates must remain inside their split.

Membership is inclusive on both interval endpoints. Fundamentals are backward-as-of joined on `available_date = datadate + 6 months`. Search context is train-only; validation selects immutable snapshots; test is opened only after formulas and weights are frozen. Validation/test evaluators must not mutate pools, model state, or candidate archives.


For `r1_oof` / `r2_paired_oof`, each internal fit and score interval independently requires its t+21 label exit to remain inside that interval. Internal score windows are ordered, disjoint training-feedback windows, not independent validation. HAC preserves purged gaps on the original trading-day axis. Fold weights are distinct from full-train outer-validation weights. See [rolling OOF protocol](rolling_oof.md).
