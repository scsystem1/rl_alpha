# Evaluation protocol

This is the executable protocol implemented by `evaluation/finalize.py`. It
separates search rewards from final evaluation and treats raw IC as diagnostic
only. Test data may be opened per method after that method has a complete,
correct-budget state, accepted artifacts, an exact current cell identity, and
compatible panel/evaluator fingerprints.

## Per-cell complete-case support

Before loading test, evaluation enumerates only the requested method cells.
Each cell then freezes two masks from its own final pool:

- Cell fit: train + validation membership, explicit trade eligibility,
  complete Balanced-22 exposure, finite label, and finite values for every
  frozen expression.
- Cell test trade: test membership, trade eligibility, complete exposure,
  and finite values for every frozen expression. It deliberately excludes the
  test label so portfolio formation cannot use label availability.
- RNIC sample: cell test trade intersected with the finite test label. Every
  signal and the label are projected on this exact same daily sample.

Metrics record valid days and observations for fit, test IC, and test trading.
Different final pools may therefore have different sample sizes; those counts
are descriptive and never block evaluation of another method.

## One fit/OOS transform

`FactorTransformPipeline` is serialized into `combiner.json`. The order is:

1. On each date, use the frozen common mask and intersect all factor values,
   the label when evaluating IC, and all exposure columns.
2. Winsorize each signal cross-section at 1%/99%, then cross-sectionally
   z-score. These are explicitly per-date operations; there are no global
   moments to re-estimate on test.
3. Residualize every signal and the label jointly against the same Balanced-22
   design and the same complete-case sample. Portfolio signal residualization
   uses the label-free trade sample.
4. Cross-sectionally z-score residuals once.
5. Fit ridge-combination weights only on transformed train + validation. The
   same frozen weights and pipeline are applied out of sample.

Daily projection diagnostics retain observation count, design rank, condition
number, status, and maximum residual exposure. Insufficient/singular dates
remain missing rather than returning a plausible zero.

## Final factor statistics

For every final-pool expression, `factor_rnic_daily.parquet` contains raw
Pearson/rank IC diagnostics, primary Pearson/rank RNIC, observations, and
projection diagnostics. `factor_significance.{parquet,csv}` contains:

- number of finite dates, mean, median, sample standard deviation, and positive
  day rate;
- Newey-West/HAC standard error and two-sided normal-approximation p-value with
  lag 20;
- a 95% moving-block bootstrap interval with block length 20, 2,000 resamples,
  and fixed seed plus the frozen cell seed;
- formula mean, final ridge-weight sign, and direction-adjusted mean;
- Benjamini-Hochberg `q_value`, separately within each final pool and metric,
  at the predeclared 5% threshold.

Formula direction and ridge-weight sign are frozen before test. Test results
never flip signs or select factors. Insufficient series receive an explicit
status and missing p/q values.

## Portfolio protocols

Both portfolios rebalance every five trading days using four 20-day sleeves.
A signal at close `t` executes at close `t+1`; the new sleeve begins earning
the return at `t+2`. One-way costs are reported at 0 and 10 bps on actual
execution turnover.

- Dollar-neutral: equal-weight top/bottom 20% of the transformed score, gross
  1 and net 0 at the sleeve target. It does not claim Balanced-22 neutrality.
- Fully neutral: the same starting target is projected by an explicit QP with
  gross 1, net 0, zero exposure to the 21 non-intercept Balanced-22 columns,
  correct long/short side, trade eligibility, and maximum absolute weight
  2%. Solver status alone is insufficient: every returned solution is
  recomputed and rejected if net exceeds `1e-8`, risk exposure `1e-6`, gross
  error `1e-6`, or weight error `1e-6`.

An infeasible fully-neutral rebalance keeps the prior sleeve and records the
failure and solver audit. A held name with a missing return is kept at its last
observable value for that day (zero return contribution); observable holdings
continue to contribute at their actual weights and are not retrospectively
renormalized. This avoids both all-NaN performance paths and look-ahead leverage.
The daily missing-name count and missing gross weight are retained, while summary
metrics report affected days, maximum missing weight, and held-return capital
coverage. A non-finite portfolio-level return still invalidates the complete
performance path. Maximum drawdown starts from wealth 1 before the first return.
Realized named exposures and missing-held-exposure days are reported for both
portfolio types.

## Primary versus diagnostic outputs

Primary factor metrics are `primary_pearson_rnic` and `rank_rnic`; primary
portfolio evidence is the transformed dollar-neutral and fully-neutral result.
`raw_ic_diagnostic` is retained to quantify neutralization retention but is
not a final claim. Cross-method comparisons use same-date paired RNIC and
paired block-resampled fully-neutral 10 bps Sharpe differences. A confidence
interval containing zero is labeled uncertain, not equivalent.

No complete new experiment matrix has yet passed this protocol. The code, CPU
semantic tests and formal Verl GRPO GPU smoke are present, but final output
schemas remain unvalidated on a full Random/GP/Base-LLM/GRPO small matrix.
