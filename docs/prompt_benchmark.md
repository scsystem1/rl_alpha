# Unified prompt contract

The shared Base-LLM / GRPO prompt is `unified_rolling_summary_v8`. It contains the full formula pool, DSL vocabulary and limits, the next-close 20-trading-day target, Balanced-22 neutralization, and compact canonical training evidence. It has no named search themes or rotating hints.

The numerical evidence is one rolling OOF RNIC plus one signed coefficient per formula. Coefficients are L1-normalized within each fit fold and then averaged; they describe how the combination uses factors, not importance. All reward variants use the same neutralized diagnostic definition, cached once per frozen pool version. Neither reward-specific pool objectives nor validation/test statistics enter the prompt.

The structured depth-six grammar is provided separately from the natural-language prompt. The contract hash includes the system text, user template, summary definition, grammar and vocabulary. Eight independent completions see an identical frozen prompt. Numeric-summary ablation is confined to the offline benchmark script, with matched pool snapshots and random seeds.

The pinned Qwen3.5-2B tokenizer and config hashes match the model configuration. Full 20-factor prompts use 763 tokens for ordinary formulas and 1813 for the tested complex legal formulas, leaving room under 4096 with 128 output tokens reserved. This validates length only. See [protocol and commands](rolling_oof.md) and [token profile](rolling_oof_token_profile.json).
