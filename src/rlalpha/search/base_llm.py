from __future__ import annotations

import base64
import os
import pickle
import random
import time
from pathlib import Path
from typing import Any

from ..dsl.parser import parse_llm_response
from .models import Candidate, CandidateOutcome, SearchContext
from .prompts import DSL_GRAMMAR, HINTS, build_messages


def configure_packaged_cuda_toolchain() -> Path | None:
    """Prefer the CUDA toolkit bundled with the locked Python environment."""
    try:
        import nvidia
    except ImportError:
        return None
    root = Path(next(iter(nvidia.__path__)))
    candidates = sorted(root.glob("cu*/bin/nvcc"), reverse=True)
    if not candidates:
        return None
    cuda_home = candidates[0].parent.parent
    os.environ["CUDA_HOME"] = str(cuda_home)
    os.environ["PATH"] = str(cuda_home / "bin") + os.pathsep + os.environ.get("PATH", "")
    os.environ.setdefault("FLASHINFER_WORKSPACE_BASE", "/tmp/rlalpha_flashinfer")
    # The host toolkit headers are older than the environment compiler; the
    # native vLLM sampler avoids an unnecessary FlashInfer JIT dependency.
    os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
    return cuda_home


def resolve_model_path(config: dict[str, Any]) -> Path:
    model = config.get("model", {})
    if model.get("path"):
        path = Path(model["path"])
    else:
        root = Path(os.environ.get("RLALPHA_MODEL_SEARCH_ROOT", "/data/shared/huggingface"))
        candidates = [root / "Qwen3.5-2B"]
        candidates.extend(path.parent for path in root.glob("**/config.json") if "qwen3.5-2b" in str(path.parent).lower())
        complete = [path for path in candidates if (path / "config.json").exists() and any(path.glob("*.safetensors"))]
        if len({path.resolve() for path in complete}) != 1:
            raise FileNotFoundError(f"expected one complete Qwen3.5-2B under {root}, found {complete}")
        path = complete[0]
    if not (path / "config.json").exists():
        raise FileNotFoundError(path)
    return path.resolve()


class BaseLLMSearcher:
    def __init__(self, seed: int, config: dict[str, Any]):
        self.rng = random.Random(seed)
        self.seed = seed
        self.config = config
        self._llm = None
        self._processor = None
        self.raw_completions = 0
        self.total_tokens = 0
        self.gpu_seconds = 0.0

    @classmethod
    def from_config(cls, seed: int, config: dict[str, Any]) -> "BaseLLMSearcher":
        return cls(seed, config)

    def _load(self) -> None:
        if self._llm is not None:
            return
        configure_packaged_cuda_toolchain()
        from transformers import AutoProcessor
        from vllm import LLM

        path = resolve_model_path(self.config)
        self._processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        self._llm = LLM(
            model=str(path),
            trust_remote_code=True,
            dtype="bfloat16",
            max_model_len=int(self.config.get("rollout", {}).get("max_model_len", 4096)),
            gpu_memory_utilization=float(os.environ.get("RLALPHA_VLLM_MEMORY_UTILIZATION", "0.18")),
            enforce_eager=True,
        )

    def propose(self, context: SearchContext, n: int) -> list[Candidate]:
        self._load()
        from vllm import SamplingParams
        from vllm.sampling_params import StructuredOutputsParams

        assert self._processor is not None and self._llm is not None
        messages = [build_messages(context, HINTS[(self.raw_completions + index) % len(HINTS)]) for index in range(n)]
        prompts = [self._processor.apply_chat_template(value, tokenize=False, add_generation_prompt=True, enable_thinking=False) for value in messages]
        structured = StructuredOutputsParams(grammar=DSL_GRAMMAR)
        params = [SamplingParams(temperature=1.0, top_p=1.0, top_k=20, presence_penalty=2.0, repetition_penalty=1.0, max_tokens=128, seed=self.rng.randrange(2**31), structured_outputs=structured) for _ in prompts]
        started = time.monotonic()
        outputs = self._llm.generate(prompts, params, use_tqdm=False)
        self.gpu_seconds += time.monotonic() - started
        candidates = []
        for output in outputs:
            text = output.outputs[0].text.strip()
            self.total_tokens += len(output.outputs[0].token_ids)
            try:
                node = parse_llm_response(text)
            except (ValueError, TypeError):
                node = None
            candidates.append(Candidate(node, "base_llm", raw_text=text))
        self.raw_completions += len(candidates)
        return candidates

    def observe(self, outcomes: list[CandidateOutcome]) -> None:
        return None

    def state_dict(self) -> dict[str, Any]:
        encoded = base64.b64encode(pickle.dumps(self.rng.getstate())).decode("ascii")
        return {"rng_state": encoded, "seed": self.seed, "raw_completions": self.raw_completions, "total_tokens": self.total_tokens, "gpu_seconds": self.gpu_seconds}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.setstate(pickle.loads(base64.b64decode(state["rng_state"])))
        self.seed = int(state["seed"])
        self.raw_completions = int(state["raw_completions"])
        self.total_tokens = int(state.get("total_tokens", 0))
        self.gpu_seconds = float(state.get("gpu_seconds", 0.0))
