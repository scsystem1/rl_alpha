from __future__ import annotations

import numpy as np


def place_scalar_rewards(rewards: np.ndarray, response_mask: np.ndarray) -> np.ndarray:
    """Place each outcome reward on its final non-padding response token."""
    rewards = np.asarray(rewards, dtype=np.float32)
    mask = np.asarray(response_mask, dtype=bool)
    if mask.shape[0] != len(rewards):
        raise ValueError("reward batch and response mask differ")
    output = np.zeros(mask.shape, dtype=np.float32)
    lengths = mask.sum(axis=1)
    for row, length in enumerate(lengths):
        if length:
            output[row, int(length) - 1] = rewards[row]
    return output


class ReadOnlyRewardBridge:
    def __init__(self, scorer, state_hash):
        self.scorer = scorer
        self.state_hash = state_hash

    def score(self, *args, **kwargs):
        before = self.state_hash()
        result = self.scorer(*args, **kwargs)
        if self.state_hash() != before:
            raise RuntimeError("read-only reward bridge mutated state")
        return result
