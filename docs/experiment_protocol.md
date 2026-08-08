# Experiment protocol

The two axes are search method (`random`, `gp`, `base_llm`, `grpo_llm`) and reward (`R0`, `R1`, `R2_LCB`). All cells share the AST, validity rules, group size eight, pool capacity 20, and valid-unique market-evaluation budget. A group is scored against one frozen pool and admits at most one candidate.

M0-M9 are implemented. Search remains train-only, validation is used only for
snapshot selection, and test is opened behind an experiment-level transaction
after every cell's formula set is frozen. Final evaluation constructs one
cross-method test universe before computing any metric.

The screening matrix is 4 methods x 3 rewards x seed 0 at 5,000 valid unique
market evaluations per cell. Confirmatory runs use the six cells in
`configs/experiment/confirmatory.yaml`, seeds 0/1/2, and 20,000 evaluations;
seed 0 resumes its screening archive. GRPO always keeps eight rollouts and
eight prompt groups per frozen-pool stage.
