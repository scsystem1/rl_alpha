from pathlib import Path
from types import SimpleNamespace

from rlalpha.search.alphasage.run import run_alphasage
from rlalpha.utils.io import write_json, write_yaml


ROOT = Path(__file__).resolve().parents[2]


def test_alphasage_adapter_uses_one_hundred_by_eight_factor_budget(tmp_path, monkeypatch):
    alphasage = tmp_path / "AlphaSAGE"
    write_json(alphasage / "train_rlalpha.py", {})
    runs = tmp_path / "runs"
    config = tmp_path / "window.yaml"
    write_yaml(config, {
        "protocol": "recent_alpha_v1",
        "paths": {
            "code_root": str(ROOT),
            "runs_root": str(runs),
        },
        "experiment": {
            "cells": [["alphasage", "r0"]],
            "seeds": [0],
            "search_steps": 100,
            "proposal_group_size": 8,
            "pool_capacity": 20,
        },
    })
    commands = []

    def execute(command, **kwargs):
        commands.append((command, kwargs))
        write_json(runs / "demo/alphasage/r0/seed_0/train_metrics.json", {
            "search_steps": 100,
            "completed_steps": 100,
            "candidates_per_step": 8,
            "factor_budget": 800,
            "raw_proposals": 800,
            "valid_unique_evaluations": 800,
        })
        return SimpleNamespace(returncode=0)

    monkeypatch.setenv("RLALPHA_ALPHASAGE_ROOT", str(alphasage))
    monkeypatch.setattr("rlalpha.search.alphasage.run.subprocess.run", execute)
    result = run_alphasage(config, "r0", 0, 100, "demo")
    command = commands[0][0]
    assert command[command.index("--raw-proposal-budget") + 1] == "800"
    assert command[command.index("--search-steps") + 1] == "100"
    assert command[command.index("--proposal-group-size") + 1] == "8"
    assert "--evaluate" not in command
    assert result["factor_budget"] == 800
