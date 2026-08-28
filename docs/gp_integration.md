# AlphaGen GP integration

The maintained `gp` method loads the modified `gplearn` implementation from
the configured read-only AlphaGen checkout. It does not retain RLAlpha's former
custom AST crossover/mutation implementation.

## Call boundary

AlphaGen owns:

- ramped half-and-half random program construction;
- tournament selection;
- crossover;
- subtree, hoist and point mutation;
- reproduction and parent indices.

RLAlpha owns only the experiment adapter:

- rendering AlphaGen programs into the shared typed DSL;
- rejecting trees outside the depth, node and lookback contract;
- rejecting expressions already emitted by the GP run;
- evaluating signals on the train panel;
- scoring all eight candidates against one frozen pool snapshot;
- returning the raw `delta_objective` as GP fitness;
- admitting at most one candidate after the complete generation;
- checkpoint, lineage and budget state.

AlphaGen's original operation weights are `30:10:1:10` for crossover,
subtree, hoist and point mutation, with the remaining probability assigned to
pure reproduction. RLAlpha preserves those four relative weights after
normalization and sets pure reproduction to zero: an unchanged parent is not a
new factor and would otherwise create a GP-only free rescore path.

One AlphaGen population is exactly one proposal group. The formal configuration
therefore fixes `population_size=8`, matching the experiment-wide
`proposal_group_size=8`. A population cannot mix pool versions: every outcome
must carry the same `pre_group_pool_version`, otherwise the searcher fails.

## Fitness and admission

For generation `t`, every valid candidate `x` receives the same score used by
the other search methods:

```text
fitness_t(x) = J(best counterfactual pool containing x) - J(P_t)
```

If the pool is not full, the counterfactual is `P_t + x`. If it is full, all
single-factor replacements are evaluated and the best is retained in the
candidate score. All counterfactuals use the unchanged `P_t`. Only after all
eight fitness values are available does `PoolManager.consider_group` admit the
best positive-delta candidate, at most once.

The completed generation becomes the parent population. Its fitness values
were all measured on one pool version and are used by AlphaGen to produce the
next generation. The offspring are then evaluated together on the next frozen
pool. There is no free `gp_rescore` path and no mixture of fitness values inside
a generation.

## Identity and resume

The checkpoint records the engine name and exact source path, operation
configuration, NumPy RNG state, serialized AlphaGen program tokens, generation,
fitness pool version and all previously generated expression hashes. Resume
rejects a different engine source or configuration and reproduces the next
generation exactly. The run manifest separately fingerprints the AlphaGen Git
checkout.
