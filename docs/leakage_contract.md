# Leakage contract

Signals at date t may read data at or before t only. `Ref(x,n)` is a backward lag. The 20-day label is the product of `DlyRet[t+2:t+22]`, corresponding to next-close entry and t+21 close exit. Exit dates must remain inside their split.

Membership is inclusive on both interval endpoints. Fundamentals are backward-as-of joined on `available_date = datadate + 6 months`. Search context is train-only; validation selects immutable snapshots; test is opened only after formulas and weights are frozen. Validation/test evaluators must not mutate pools, model state, or candidate archives.

