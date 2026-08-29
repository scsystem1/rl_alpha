from __future__ import annotations

from pathlib import Path
from typing import Any, Callable
import builtins

import torch
from verl.utils.dataset.rl_dataset import RLHFDataset


if not hasattr(builtins, "_RLAlphaOnlineTrainingComplete"):
    class _OnlineTrainingComplete(RuntimeError):
        """Internal control signal emitted after the final optimizer update."""

    builtins._RLAlphaOnlineTrainingComplete = _OnlineTrainingComplete
OnlineTrainingComplete = builtins._RLAlphaOnlineTrainingComplete


if not hasattr(builtins, "_rlalpha_online_dataset_callbacks"):
    builtins._rlalpha_online_dataset_callbacks = {}
_CALLBACKS: dict[str, Callable[[dict[str, Any]], dict[str, Any] | None]] = (
    builtins._rlalpha_online_dataset_callbacks
)


def register_online_callback(
    data_file: str | Path,
    callback: Callable[[dict[str, Any]], dict[str, Any] | None],
) -> str:
    key = str(Path(data_file).resolve())
    if key in _CALLBACKS:
        raise RuntimeError(f"online dataset callback already registered for {key}")
    _CALLBACKS[key] = callback
    return key


def unregister_online_callback(data_file: str | Path) -> None:
    _CALLBACKS.pop(str(Path(data_file).resolve()), None)


class OnlinePoolDataset(RLHFDataset):
    """Verl dataset whose next prompt is committed after the prior update."""

    @staticmethod
    def _build_messages(example: dict[str, Any], key: str) -> list[dict[str, Any]]:
        """Text-only compatibility shim for the supported Verl releases.

        Some Verl workers dynamically load the custom class against an older
        RLHFDataset that does not expose `_build_messages`.  RLAlpha prompts do
        not contain multimodal placeholders, so returning a defensive copy is
        the exact required behavior on both APIs.
        """
        return [dict(message) for message in example[key]]

    @staticmethod
    def _process_multi_modal_info(
        messages: list[dict[str, Any]],
        image_patch_size: int,
        config: Any,
    ) -> tuple[None, None, None]:
        """Return empty modalities for RLAlpha's deliberately text-only prompts."""
        return None, None, None

    @classmethod
    async def process_multi_modal_info(
        cls,
        messages: list[dict[str, Any]],
        image_patch_size: int,
        config: Any,
    ) -> tuple[None, None, None]:
        return None, None, None

    def __init__(self, *args: Any, **kwargs: Any):
        super().__init__(*args, **kwargs)
        self.callback_key = str(Path(str(self.original_data_files[0])).resolve())
        # Verl applies data.custom_cls to both train and validation datasets.
        # Only the online training parquet has a registered transition
        # callback; validation remains an ordinary immutable dataset.
        self.is_online = self.callback_key in _CALLBACKS
        self._rows = [dict(self.dataframe[index]) for index in range(len(self.dataframe))]
        self._completed_batches = 0

    def __len__(self) -> int:
        return len(self._rows) if hasattr(self, "_rows") else super().__len__()

    def __getitem__(self, item: int) -> dict[str, Any]:
        row_dict = dict(self._rows[item])
        row_dict["raw_prompt"] = self._build_messages(row_dict, key=self.prompt_key)
        row_dict.pop(self.image_key, None)
        row_dict.pop(self.video_key, None)
        row_dict.pop(self.audio_key, None)
        row_dict["dummy_tensor"] = torch.tensor([0], dtype=torch.uint8)
        if "extra_info" not in row_dict or row_dict["extra_info"] is None:
            row_dict["extra_info"] = {}
        extra = row_dict["extra_info"]
        row_dict["index"] = extra.get("index", 0)
        row_dict["tools_kwargs"] = extra.get("tools_kwargs", {})
        row_dict["interaction_kwargs"] = extra.get("interaction_kwargs", {})
        return row_dict

    def on_batch_end(self, batch: Any) -> None:
        if not self.is_online:
            return
        callback = _CALLBACKS[self.callback_key]
        next_row = callback({"batch": batch})
        if next_row is None:
            raise OnlineTrainingComplete("fixed search-step budget exhausted")
        self._completed_batches += 1
        if self._completed_batches >= len(self._rows):
            raise OnlineTrainingComplete("online dataset capacity exhausted")
        # StatefulDataLoader may restore a sampler offset. Fill every unread
        # placeholder so the next requested index observes the committed pool.
        for index in range(self._completed_batches, len(self._rows)):
            self._rows[index] = next_row
