# Repository audit

AlphaGen is used only as a design reference. Its typed expression tree and parser live in `alphagen/alphagen/data/expression.py` and `parser.py`; IC interfaces are in `calculator.py`; pool behavior is in `models/linear_alpha_pool.py`; the GP entry point is `gp.py`; generic data adapters are under `alphagen_generic/`. The checkout has no top-level LICENSE file, so no AlphaGen source has been copied.

QuantEvolver is MIT licensed. Its prompt parquet schema is in `quant_evolver/rft/dataset.py`, no-think handling in `no_think_dataset.py`, token reward bridge in `reward_bridge.py`, Verl wiring in `verl_main.py`, cross-sectional RankIC in `evaluation/cross_sectional_rankic.py`, and its independent DSL under `quant_evolver/dsl/`.

Required adaptations are material: this project removes VWAP and future-reference targets; uses historical S&P membership and daily CRSP CIZ data; adds a point-in-time accounting lag; shares one AST across all proposers; calculates pool delta against a frozen pool; separates read-only validation; uses Pearson risk-neutral IC and HAC LCB; and sets Qwen3.5 `use_remove_padding: false`. QuantEvolver currently passes the same reward bridge to training and validation and hard-codes `use_remove_padding: true`, so those pieces cannot be reused unchanged.

