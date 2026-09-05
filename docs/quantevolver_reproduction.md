# QuantEvolver fair reproduction

This adapter reproduces the public QuantEvolver training design while using
RLAlpha's audited S&P 500 panel, split contract, factor DSL, Qwen3.5-2B model,
and final evaluator. The public checkout remains unmodified at commit
`4eb0e78842138ada5334349585b114ad923564e8`.

## Fixed comparison contract

- Miner: `/data/shared/huggingface/Qwen3.5-2B`, pinned by the same revision and
  hashes as Base-LLM/formal GRPO.
- Budget: 250 optimizer updates per seed, one prompt per update, exactly eight
  completions per prompt, seeds 0/1/2.
- Search data: RLAlpha train split only (2010-2018). QuantEvolver's task-bank
  idea is retained by crossing eight public-seed adaptations with early,
  middle, late, and full train regimes.
- Action space: the shared typed RLAlpha DSL and structured-decoding grammar.
- Reward: QuantEvolver-native mean daily cross-sectional RankIC plus exact,
  structural-family, and behavioral diversity/complementarity shaping. This is
  labeled `qe_native`; it is not relabeled as R0/R1/R2.
- Mined factor database: executable factors with RankIC >= 0.01 and coverage
  >= 0.60 are accumulated without reading validation or test.
- Selection: validation RankIC ranking followed by greedy absolute-correlation
  filtering at 0.70, capped at 20 factors. Test remains unopened until all
  three cells are complete and frozen.
- Evaluation: the same neutralized RNIC, portfolio, turnover/cost, bootstrap,
  HAC, and reporting implementation used by every RLAlpha method.

## Environment and launch

The independent environment is `.venvs/quantevolver`. It inherits the already
verified CUDA/Verl stack from the `rlalpha` conda environment and installs only
editable `quant-evolver` and `rlalpha` packages inside the venv.

Run the resource-aware pipeline with:

```bash
ours/scripts/run_quantevolver_fair_250.sh
```

It first runs a real two-update smoke, then the three 250-update cells, final
evaluation, and report. A cell starts when physical GPU 4 has at least
20 GiB free. The smoke and all three independent seeds run sequentially on
GPU 4; physical GPU 0 is never used by this reproduction.
