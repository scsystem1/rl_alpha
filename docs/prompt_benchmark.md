# Unified prompt contract

RLAlpha maintains exactly one prompt, `unified_compact_v7`, for both Base-LLM
and formal GRPO. There is no prompt-version switch and no rotating hint. Reward
variants also do not change the prompt.

The prompt contains only:

1. every formula currently in the frozen train-only pool;
2. the allowed features, operators, windows, and constants;
3. one short goal: propose a valid, non-duplicate complement that improves the
   current pool on training data;
4. one compact hint line covering momentum, mean reversion, volatility,
   price-volume interaction, multi-horizon structure, and cross-sectional
   ranking;
5. the exact `<expr>FORMULA</expr>` output schema.

The finite depth-six, featureful grammar is enforced by structured decoding and
included in the prompt contract hash, but is not repeated in the
natural-language prompt.
Parsing, canonicalization, semantic validity, duplicate detection, and
train-only market evaluation remain mandatory after decoding.

## Sampling and admission protocol

- Base-LLM: render the prompt once, sample it independently eight times, score
  all eight against one frozen pool, and admit at most the best one.
- GRPO: use one prompt row with `rollout.n=8`; the eight rewards form one GRPO
  group and one optimizer update. After the update, admit at most the best one
  against the same frozen pre-round pool.
- Random follows the same eight-candidate frozen-pool admission round. GP is
  maintained separately.

Run the tokenizer profile for the sole prompt with:

```bash
python scripts/benchmark_prompts.py \
  --model /data/shared/huggingface/Qwen3.5-2B \
  --output /data/sunyuxiang/rl_alpha/runs/prompt_benchmark/token_profile.json
```

The profile reports token counts at representative pool sizes; it does not
select among templates because alternative prompt versions no longer exist.
