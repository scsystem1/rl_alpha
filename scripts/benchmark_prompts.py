from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rlalpha.search.models import SearchContext
from rlalpha.search.prompts import build_messages, prompt_contract
from rlalpha.utils.io import write_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Token profile for the one shared RLAlpha prompt.")
    parser.add_argument("--model", default="/data/shared/huggingface/Qwen3.5-2B")
    parser.add_argument("--output", default="artifacts/prompt_benchmark/token_profile.json")
    args = parser.parse_args()
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True, trust_remote_code=True)
    formulas = (
        "Mean($return,20)", "Delta($close,20)", "Std($return,60)", "Corr($close,$volume,20)",
        "CSRank(Mean($return,5))", "Div(Mean($return,20),Add(Std($return,20),0.01))",
        "Sub(EMA($close,10),EMA($close,40))", "TSRank($volume,20)", "Mad($return,20)",
        "Corr($return,$volume,5)", "CSZScore(Delta($close,5))", "Div(Delta($close,20),Add(Std($close,20),0.01))",
    )
    rows = []
    for pool_size in (0, 5, 12):
        context = SearchContext(3, formulas[:pool_size], tuple(np.linspace(-0.3, 0.4, pool_size)), 0.0123, 400, 5000)
        messages = build_messages(context)
        rendered = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        tokens = processor.tokenizer(rendered, add_special_tokens=False)["input_ids"]
        rows.append({"pool_size": pool_size, "input_tokens": len(tokens), "characters": len(rendered)})
    payload = {"status": "single_shared_prompt", "model_path": str(Path(args.model).resolve()), "contract": prompt_contract(), "rows": rows}
    write_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
