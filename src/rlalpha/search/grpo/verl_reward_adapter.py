from __future__ import annotations

import json
import math
import threading
from pathlib import Path
from typing import Any, Callable

import numpy as np


class VerlRLAlphaRewardBridge:
    """QuantEvolver/Verl-compatible, frozen-state RLAlpha reward adapter.

    ``score_texts`` returns one diagnostic mapping per decoded completion.  It
    may read a frozen pool, but the supplied ``state_hash`` must remain stable
    for the complete reward call.  Validation uses the same class without an
    archive path, making it mechanically read-only.
    """

    def __init__(
        self,
        tokenizer: Any,
        score_texts: Callable[[list[str], list[dict[str, Any]]], list[dict[str, Any]]],
        state_hash: Callable[[], str],
        *,
        archive_path: str | Path | None = None,
        invalid_penalty: float = -1.0,
    ):
        self.tokenizer = tokenizer
        self.score_texts = score_texts
        self.state_hash = state_hash
        self.archive_path = None if archive_path is None else Path(archive_path)
        self.invalid_penalty = float(invalid_penalty)
        self._archive_lock = threading.Lock()

    def _decode(self, data: Any) -> tuple[list[str], np.ndarray]:
        responses = data.batch["responses"]
        response_mask = data.batch.get("response_mask")
        if response_mask is None:
            prompt_length = data.batch["prompts"].shape[-1]
            response_mask = data.batch["attention_mask"][:, prompt_length:]
        mask = response_mask.detach().cpu().numpy().astype(bool)
        texts = []
        for row in range(len(responses)):
            ids = responses[row][response_mask[row].bool()]
            texts.append(self.tokenizer.decode(ids.detach().cpu().tolist(), skip_special_tokens=True))
        return texts, mask

    @staticmethod
    def _extra_info(data: Any, count: int) -> list[dict[str, Any]]:
        non_tensor = getattr(data, "non_tensor_batch", {}) or {}
        raw = non_tensor.get("extra_info")
        if raw is None:
            return [{} for _ in range(count)]
        return [dict(raw[index]) if index < len(raw) and isinstance(raw[index], dict) else {} for index in range(count)]

    def _archive(self, records: list[dict[str, Any]]) -> None:
        if self.archive_path is None:
            return
        self.archive_path.parent.mkdir(parents=True, exist_ok=True)
        with self._archive_lock:
            with self.archive_path.open("a", encoding="utf-8") as handle:
                for record in records:
                    handle.write(json.dumps(record, sort_keys=True, default=str) + "\n")
                handle.flush()

    def __call__(self, data: Any, return_dict: bool = False):
        import torch

        before = self.state_hash()
        texts, response_mask = self._decode(data)
        extra_info = self._extra_info(data, len(texts))
        records = self.score_texts(texts, extra_info)
        if len(records) != len(texts):
            raise RuntimeError("reward adapter returned the wrong sample count")
        if self.state_hash() != before:
            raise RuntimeError("reward adapter mutated its frozen pool/model state")
        rewards = []
        normalized_records = []
        for text, info, record in zip(texts, extra_info, records, strict=True):
            value = record.get("shaped_reward", self.invalid_penalty)
            value = float(value) if value is not None else self.invalid_penalty
            if not math.isfinite(value):
                value = self.invalid_penalty
                record = {**record, "valid": False, "reason_code": "non_finite_reward"}
            value = float(np.clip(value, -1.0, 1.0))
            rewards.append(value)
            normalized_records.append({**info, **record, "raw_text": text, "shaped_reward": value, "frozen_state_hash": before})
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        lengths = response_mask.sum(axis=1)
        for row, length in enumerate(lengths):
            if length > 0:
                reward_tensor[row, int(length) - 1] = rewards[row]
        self._archive(normalized_records)
        if not return_dict:
            return reward_tensor
        keys = sorted({key for record in normalized_records for key in record})
        dense_extra = {key: [record.get(key) for record in normalized_records] for key in keys}
        return {"reward_tensor": reward_tensor, "reward_extra_info": dense_extra}
