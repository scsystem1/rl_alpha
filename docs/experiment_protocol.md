# Experiment protocol

The two axes are search method (`random`, `gp`, `base_llm`, `grpo_llm`) and reward (`R0`, `R1`, `R2_LCB`). All cells share the AST, validity rules, group size eight, pool capacity 20, and valid-unique market-evaluation budget. A group is scored against one frozen pool and admits at most one candidate.

This first delivery implements milestones M0–M4. Large search runs are deliberately disabled. M5–M9 and GPU/GRPO smoke tests begin only after CPU/data/DSL/reward acceptance and explicit compute-budget confirmation.

