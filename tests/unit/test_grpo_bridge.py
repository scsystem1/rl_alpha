import numpy as np

from rlalpha.search.grpo.reward_bridge import ReadOnlyRewardBridge, place_scalar_rewards


def test_reward_is_placed_on_last_non_pad_token() -> None:
    mask = np.array([[1, 1, 0, 0], [1, 1, 1, 0]], dtype=bool)
    placed = place_scalar_rewards(np.array([0.5, -1.0]), mask)
    assert np.array_equal(placed, np.array([[0, 0.5, 0, 0], [0, 0, -1, 0]], dtype=np.float32))


def test_validation_reward_bridge_is_read_only() -> None:
    state = {"version": 1}
    bridge = ReadOnlyRewardBridge(lambda value: value + 1, lambda: state["version"])
    assert bridge.score(2) == 3
