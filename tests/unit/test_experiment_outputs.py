from pathlib import Path

from rlalpha.utils.experiment_log import append_event, write_result_summary
from rlalpha.utils.io import write_json, write_yaml


def test_result_writers_do_not_leave_lock_files(tmp_path):
    write_json(tmp_path / "progress.json", {"status": "running"})
    write_yaml(tmp_path / "effective_config.yaml", {"group_size": 8})
    append_event(tmp_path / "experiment.log", "round_complete", round=1, generated=8)
    write_result_summary(
        tmp_path / "result.md",
        experiment_id="demo",
        method="base_llm",
        reward="r0",
        seed=0,
        budget=8,
        ledger={"raw_proposals": 8, "valid_unique_evaluations": 8},
        pool_version=1,
        train_objective=0.1,
        validation_objective=0.05,
        expressions=["Mean($return,20)"],
    )
    assert not list(tmp_path.rglob("*.lock"))
    assert "generated=8" in (tmp_path / "experiment.log").read_text(encoding="utf-8")
    assert "Mean($return,20)" in (tmp_path / "result.md").read_text(encoding="utf-8")


def test_repository_result_tree_has_no_lock_artifacts(tmp_path):
    # This guards the public output contract rather than implementation detail:
    # all synchronization files must live outside a run directory.
    write_json(tmp_path / "nested" / "metrics.json", {"ok": True})
    assert [path for path in tmp_path.rglob("*") if path.name.endswith(".lock")] == []
