from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np

from rlalpha.dsl.grammar import sample_ast
from rlalpha.search.models import SearchContext, TrainPoolSummary
from rlalpha.search.prompts import build_messages, prompt_contract
from rlalpha.utils.io import write_json


def profile_prompts(tokenizer, max_model_len=4096, response_length=128):
    ordinary = tuple(f"CSRank(Mean({feature},{window}))" for feature in
        ("$open", "$high", "$low", "$close", "$volume") for window in (5, 10, 20, 60))
    rng = random.Random(581)
    generated = {sample_ast(rng, 6).canonical() for _ in range(3000)}
    complex_pool = tuple(sorted(generated, key=lambda f: (len(tokenizer.encode(f, add_special_tokens=False)), f), reverse=True)[:20])
    rows = []
    for kind, pool in (("ordinary", ordinary), ("complex_legal", complex_pool)):
        for size in ((0, 5, 12, 20) if kind == "ordinary" else (20,)):
            weights = np.linspace(-.3, .4, size)
            weights /= max(float(np.abs(weights).sum()), 1e-30)
            context = SearchContext(3, pool[:size], (), .0123, 400, 2000,
                prompt_summary=TrainPoolSummary(.0123, tuple(weights)))
            rendered = tokenizer.apply_chat_template(build_messages(context), tokenize=False,
                add_generation_prompt=True, enable_thinking=False)
            tokens = tokenizer.encode(rendered, add_special_tokens=False)
            rows.append({"kind": kind, "pool_size": size, "input_tokens": len(tokens),
                "characters": len(rendered), "remaining_after_response": max_model_len-response_length-len(tokens),
                "formulas": list(pool[:size])})
    return {"status": "pass" if all(r["remaining_after_response"] >= 0 for r in rows) else "fail",
        "contract": prompt_contract(), "max_model_len": max_model_len, "response_length": response_length,
        "summary_values": "synthetic formatting fixture, not measured market performance", "rows": rows}


def main():
    parser = argparse.ArgumentParser(description="Text-only Qwen tokenizer profile, including a full complex pool.")
    parser.add_argument("--model", default="/data/shared/huggingface/Qwen3.5-2B")
    parser.add_argument("--output", default="/data/sunyuxiang/rl_alpha/runs/prompt_benchmark/token_profile.json")
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--response-length", type=int, default=128)
    args = parser.parse_args()
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    payload = profile_prompts(tokenizer, args.max_model_len, args.response_length)
    payload["model_path"] = str(Path(args.model).resolve())
    payload["file_hashes"] = {name: hashlib.sha256((Path(args.model)/name).read_bytes()).hexdigest()
        for name in ("tokenizer.json", "config.json", "chat_template.jinja") if (Path(args.model)/name).exists()}
    write_json(args.output, payload)
    print(json.dumps({"status": payload["status"], "file_hashes": payload["file_hashes"],
        "rows": [{k: v for k, v in row.items() if k != "formulas"} for row in payload["rows"]]}, indent=2))
    if payload["status"] != "pass":
        raise SystemExit("prompt exceeds the configured input budget; formulas must not be truncated")


if __name__ == "__main__":
    main()
