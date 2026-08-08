from __future__ import annotations

import pytest


@pytest.mark.gpu
def test_grpo_two_update_smoke_is_run_by_acceptance_script() -> None:
    pytest.skip("Run scripts/smoke_grpo.py on an acceptance GPU; this marker documents the contract.")
