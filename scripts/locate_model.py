from __future__ import annotations

import argparse
import json
from pathlib import Path


REQUIRED_TOKENIZER_FILES = {"tokenizer.json", "tokenizer_config.json"}


def candidates(root: Path) -> list[Path]:
    found = []
    for config in root.glob("**/config.json"):
        path = config.parent
        name = str(path).lower()
        if "qwen3.5" not in name or "2b" not in name:
            continue
        names = {item.name for item in path.iterdir() if item.is_file()}
        tokenizer = bool(names & REQUIRED_TOKENIZER_FILES)
        weights = any(name.endswith(".safetensors") for name in names)
        index_ok = "model.safetensors.index.json" in names or weights
        if tokenizer and index_ok:
            found.append(path.resolve())
    return sorted(set(found))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/data/shared/huggingface"))
    args = parser.parse_args()
    resolved = candidates(args.root)
    print(json.dumps({"resolved": str(resolved[0]) if len(resolved) == 1 else None, "candidates": list(map(str, resolved)), "unique": len(resolved) == 1}, indent=2))
    raise SystemExit(0 if len(resolved) == 1 else 2)

