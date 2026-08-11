from __future__ import annotations

import base64
import json
import os
import pickle
import random
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np

from ...dsl.parser import parse_llm_response
from ...utils.io import atomic_write_text, write_json
from ..base_llm import configure_packaged_cuda_toolchain, resolve_model_path
from ..models import Candidate, CandidateOutcome, SearchContext
from ..prompts import DSL_GRAMMAR, build_messages
from .reward_bridge import place_scalar_rewards


class StagedGRPOSearcher:
    """Legacy group-normalized REINFORCE implementation; not formal GRPO.

    Kept temporarily only to make old checkpoints auditable.  Formal search
    dispatch never instantiates this class; use the QuantEvolver/Verl adapter.
    """

    admission_group_interval = 8
    rollout_group = 8

    def __init__(self, seed: int, config: dict[str, Any]):
        self.seed, self.config = seed, config
        self.rng = random.Random(seed)
        self.run_dir = Path(config["run_dir"])
        self.stage = 0
        self.updates = 0
        self.groups_in_stage = 0
        self.pool_version = -1
        self.zero_group_variance = 0
        self.total_tokens = 0
        self.gpu_seconds = 0.0
        self._model = self._tokenizer = self._optimizer = self._scheduler = None
        self._compiled_grammar = None
        self._pending: dict[str, Any] | None = None
        self._restore_checkpoint: Path | None = None

    @classmethod
    def from_config(cls, seed: int, config: dict[str, Any]) -> "StagedGRPOSearcher":
        return cls(seed, config)

    def _load(self) -> None:
        if self._model is not None:
            return
        configure_packaged_cuda_toolchain()
        import torch
        from peft import LoraConfig, PeftModel, get_peft_model
        from transformers import AutoModelForImageTextToText, AutoProcessor

        path = resolve_model_path(self.config)
        processor = AutoProcessor.from_pretrained(path, local_files_only=True, trust_remote_code=True)
        self._tokenizer = processor.tokenizer
        model = AutoModelForImageTextToText.from_pretrained(path, local_files_only=True, trust_remote_code=True, dtype=torch.bfloat16, attn_implementation="sdpa")
        if self._restore_checkpoint is not None:
            model = PeftModel.from_pretrained(model, self._restore_checkpoint / "adapter", is_trainable=True)
        else:
            model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, target_modules="all-linear", lora_dropout=0.0, bias="none", task_type="CAUSAL_LM"))
        model.gradient_checkpointing_enable()
        model.to("cuda")
        import xgrammar as xgr

        text_config = getattr(model.config, "text_config", model.config)
        tokenizer_info = xgr.TokenizerInfo.from_huggingface(self._tokenizer, vocab_size=text_config.vocab_size)
        self._compiled_grammar = xgr.GrammarCompiler(tokenizer_info).compile_grammar(DSL_GRAMMAR)
        self._model = model
        parameters = [value for value in model.parameters() if value.requires_grad]
        self._optimizer = torch.optim.AdamW(parameters, lr=1e-6)
        self._scheduler = torch.optim.lr_scheduler.LambdaLR(self._optimizer, lambda _: 1.0)
        if self._restore_checkpoint is not None:
            state = torch.load(self._restore_checkpoint / "trainer_state.pt", map_location="cpu", weights_only=False)
            self._optimizer.load_state_dict(state["optimizer"])
            self._scheduler.load_state_dict(state["scheduler"])
            torch.set_rng_state(state["torch_rng"])
            torch.cuda.set_rng_state(state["cuda_rng"])

    def propose(self, context: SearchContext, n: int) -> list[Candidate]:
        if n != self.rollout_group:
            raise ValueError("GRPO requires rollout group=8")
        if self._pending is not None:
            raise RuntimeError("observe must follow each GRPO proposal group")
        self._load()
        import torch

        assert self._tokenizer is not None and self._model is not None and self._compiled_grammar is not None
        if self.pool_version != context.pool_version:
            self.pool_version = context.pool_version
            self.groups_in_stage = 0
        messages = build_messages(context)
        prompt = self._tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
        encoded = self._tokenizer(prompt, return_tensors="pt").to("cuda")
        torch.cuda.manual_seed(self.rng.randrange(2**31))
        self._model.eval()
        prompt_length = encoded["input_ids"].shape[1]

        class PresencePenalty:
            def __call__(self, input_ids, scores):
                for row in range(len(input_ids)):
                    seen = torch.unique(input_ids[row, prompt_length:])
                    if len(seen):
                        scores[row, seen] -= 2.0
                return scores

        class StopAfterGrammar:
            def __init__(self, grammar_processor, eos_token_id):
                self.grammar_processor = grammar_processor
                self.eos_token_id = eos_token_id

            def __call__(self, input_ids, scores):
                if self.grammar_processor.matchers:
                    for row, matcher in enumerate(self.grammar_processor.matchers):
                        if matcher.is_terminated():
                            scores[row].fill_(float("-inf"))
                            scores[row, self.eos_token_id] = 0.0
                return scores

        started = time.monotonic()
        microbatch = max(1, min(n, int(os.getenv("RLALPHA_GRPO_MICROBATCH", str(n)))))
        pad_token_id = self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id
        with torch.no_grad():
            batches = []
            for start in range(0, n, microbatch):
                from xgrammar.contrib.hf import LogitsProcessor as GrammarLogitsProcessor

                size = min(microbatch, n - start)
                grammar_processor = GrammarLogitsProcessor(self._compiled_grammar)
                stop_processor = StopAfterGrammar(grammar_processor, self._tokenizer.eos_token_id)
                batches.append(self._model.generate(**encoded, do_sample=True, temperature=1.0, top_p=1.0, top_k=20, repetition_penalty=1.0, logits_processor=[PresencePenalty(), grammar_processor, stop_processor], max_new_tokens=128, num_return_sequences=size, pad_token_id=pad_token_id, eos_token_id=self._tokenizer.eos_token_id))
            width = max(batch.shape[1] for batch in batches)
            generated = torch.cat([torch.nn.functional.pad(batch, (0, width - batch.shape[1]), value=pad_token_id) for batch in batches])
        self.gpu_seconds += time.monotonic() - started
        response_ids = generated[:, prompt_length:]
        self.total_tokens += int(response_ids.ne(pad_token_id).sum().item())
        texts = self._tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        candidates = []
        for text in texts:
            try:
                node = parse_llm_response(text.strip())
            except (ValueError, TypeError):
                node = None
            candidates.append(Candidate(node, "grpo_llm", raw_text=text.strip()))
        response_mask = response_ids.ne(pad_token_id)
        self._pending = {"generated": generated.detach(), "prompt_length": prompt_length, "response_mask": response_mask.detach(), "pool_version": context.pool_version, "candidates": candidates}
        return candidates

    def observe(self, outcomes: list[CandidateOutcome]) -> None:
        if self._pending is None:
            raise RuntimeError("no pending GRPO rollout")
        if len(outcomes) != self.rollout_group or self._pending["pool_version"] != self.pool_version:
            raise RuntimeError("GRPO group or frozen pool mismatch")
        import torch
        from verl.trainer.ppo.core_algos import compute_grpo_outcome_advantage

        assert self._model is not None and self._optimizer is not None and self._scheduler is not None
        rewards = np.asarray([outcome.shaped_reward for outcome in outcomes], dtype=np.float32)
        response_mask = self._pending["response_mask"]
        token_rewards = torch.as_tensor(place_scalar_rewards(rewards, response_mask.cpu().numpy()), device="cuda")
        advantages, _ = compute_grpo_outcome_advantage(token_rewards, response_mask.float(), np.zeros(len(outcomes), dtype=np.int64))
        if float(np.std(rewards)) <= 1e-12:
            self.zero_group_variance += 1
        generated = self._pending["generated"]
        prompt_length = int(self._pending["prompt_length"])
        attention = generated.ne(self._tokenizer.pad_token_id if self._tokenizer.pad_token_id is not None else self._tokenizer.eos_token_id)
        self._model.train()
        started = time.monotonic()
        targets = generated[:, prompt_length:]
        mask = response_mask.float()
        self._optimizer.zero_grad(set_to_none=True)
        microbatch = max(1, min(self.rollout_group, int(os.getenv("RLALPHA_GRPO_MICROBATCH", str(self.rollout_group)))))
        loss_value = 0.0
        for start in range(0, self.rollout_group, microbatch):
            stop = min(start + microbatch, self.rollout_group)
            output = self._model(input_ids=generated[start:stop], attention_mask=attention[start:stop], use_cache=False)
            logits = output.logits[:, prompt_length - 1 : -1].float()
            log_probs = torch.log_softmax(logits, dim=-1).gather(-1, targets[start:stop].unsqueeze(-1)).squeeze(-1)
            sequence_objective = (log_probs * advantages[start:stop] * mask[start:stop]).sum(dim=1) / mask[start:stop].sum(dim=1).clamp_min(1)
            chunk_loss = -sequence_objective.sum() / self.rollout_group
            chunk_loss.backward()
            loss_value += float(chunk_loss.detach().cpu())
        torch.nn.utils.clip_grad_norm_([value for value in self._model.parameters() if value.requires_grad], 1.0)
        self._optimizer.step()
        self._scheduler.step()
        self.gpu_seconds += time.monotonic() - started
        self.updates += 1
        self.groups_in_stage += 1
        archive = self.run_dir / "checkpoints" / f"stage_{self.stage:04d}"
        archive.mkdir(parents=True, exist_ok=True)
        write_json(archive / f"group_{self.groups_in_stage:02d}.json", {"pool_version": self.pool_version, "rewards": rewards.tolist(), "loss": loss_value, "microbatch": microbatch, "outcomes": [item.to_dict() for item in outcomes]})
        self._pending = None
        self._save_checkpoint(archive)
        if self.groups_in_stage == self.admission_group_interval:
            self.stage += 1

    def _save_checkpoint(self, directory: Path) -> None:
        import torch

        assert self._model is not None and self._optimizer is not None and self._scheduler is not None
        adapter = directory / "adapter"
        self._model.save_pretrained(adapter)
        torch.save({"optimizer": self._optimizer.state_dict(), "scheduler": self._scheduler.state_dict(), "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state()}, directory / "trainer_state.pt")
        write_json(directory / "stage_state.json", {"stage": self.stage, "updates": self.updates, "pool_version": self.pool_version, "zero_group_variance": self.zero_group_variance, "tokens": self.total_tokens, "gpu_seconds": self.gpu_seconds, "physical_gpu": os.getenv("RLALPHA_PHYSICAL_GPU"), "rollout_microbatch": int(os.getenv("RLALPHA_GRPO_MICROBATCH", str(self.rollout_group)))})
        if self.groups_in_stage >= self.admission_group_interval:
            snapshot = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free,utilization.gpu", "--format=csv,noheader,nounits"], capture_output=True, text=True, check=False)
            atomic_write_text(directory / "gpu-boundary.csv", snapshot.stdout or snapshot.stderr)

    def state_dict(self) -> dict[str, Any]:
        encoded = base64.b64encode(pickle.dumps(self.rng.getstate())).decode("ascii")
        checkpoint_stage = self.stage - 1 if self.groups_in_stage >= self.admission_group_interval else self.stage
        checkpoint = self.run_dir / "checkpoints" / f"stage_{checkpoint_stage:04d}" if self.updates else None
        return {"rng_state": encoded, "seed": self.seed, "stage": self.stage, "updates": self.updates, "groups_in_stage": self.groups_in_stage, "pool_version": self.pool_version, "zero_group_variance": self.zero_group_variance, "total_tokens": self.total_tokens, "gpu_seconds": self.gpu_seconds, "checkpoint": str(checkpoint) if checkpoint and checkpoint.exists() else None}

    def load_state_dict(self, state: dict[str, Any]) -> None:
        self.rng.setstate(pickle.loads(base64.b64decode(state["rng_state"])))
        self.stage, self.updates = int(state["stage"]), int(state["updates"])
        self.groups_in_stage, self.pool_version = int(state["groups_in_stage"]), int(state["pool_version"])
        self.zero_group_variance = int(state.get("zero_group_variance", 0))
        self.total_tokens = int(state.get("total_tokens", 0))
        self.gpu_seconds = float(state.get("gpu_seconds", 0.0))
        self._restore_checkpoint = Path(state["checkpoint"]) if state.get("checkpoint") else None
