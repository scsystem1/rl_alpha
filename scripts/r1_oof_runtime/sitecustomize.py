"""Narrow recovery hooks enabled only by the r1_oof formal launcher.

This module is imported automatically by Python because the launcher prepends
this directory to PYTHONPATH.  Nothing is patched unless
RLALPHA_R1_OOF_RUNTIME_HOOKS=1, which the launcher sets only for GRPO cells.
"""

from __future__ import annotations

import fcntl
import json
import os
from pathlib import Path


if os.getenv("RLALPHA_R1_OOF_RUNTIME_HOOKS") == "1":
    # datasets treats num_proc=1 as a child-process pool.  Serial execution is
    # equivalent for one worker and avoids pickling tokenizer mmap state after
    # Ray has initialized.
    from datasets import Dataset

    _original_filter = Dataset.filter

    def _serial_filter(self, *args, **kwargs):
        if kwargs.get("num_proc") == 1:
            kwargs = dict(kwargs)
            kwargs["num_proc"] = None
        return _original_filter(self, *args, **kwargs)

    Dataset.filter = _serial_filter

    from rlalpha.config import load_paths
    from rlalpha.search import run as _search_run_module
    from rlalpha.search.grpo.stage_coordinator import VerlGRPOStageCoordinator

    _original_run_cell = VerlGRPOStageCoordinator.run_cell

    def _resume_or_run_cell(self):
        # A terminal actor/domain checkpoint can survive interruption before
        # validation and final-pool serialization.  Resume only that remaining
        # CPU-side finalization instead of repeating an optimizer update.
        if self.updates >= self.max_training_steps:
            if self.checkpoint is None or not (self.checkpoint / "actor").is_dir():
                raise RuntimeError("completed GRPO resume is missing its paired actor checkpoint")
            completed_snapshot_ids: set[str] = set()
            snapshots_path = self.run_dir / "checkpoints" / "snapshots.jsonl"
            if snapshots_path.exists():
                for line in snapshots_path.read_text(encoding="utf-8").splitlines():
                    if not line.strip():
                        continue
                    record = json.loads(line)
                    if record.get("snapshot_id") and isinstance(record.get("validation"), dict):
                        completed_snapshot_ids.add(str(record["snapshot_id"]))
            pending = [
                snapshot
                for snapshot in self.pool_snapshots
                if str(snapshot.get("snapshot_id")) not in completed_snapshot_ids
            ]
            return {
                "updates": self.updates,
                "checkpoint": str(self.checkpoint),
                "pool_snapshots": pending,
                "metrics_path": str(self.run_dir / "grpo_session" / "verl_metrics.jsonl"),
            }
        return _original_run_cell(self)

    VerlGRPOStageCoordinator.run_cell = _resume_or_run_cell

    _original_run_search = _search_run_module.run_search

    def _locked_run_search(config_path, method, reward, seed, steps, experiment_id, resume=True):
        paths = load_paths(config_path)
        run_dir = paths.runs_root / experiment_id / method / reward / f"seed_{seed}"
        lock_path = run_dir / "scheduler" / "search.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            result_path = run_dir / "result.json"
            if result_path.exists() and (run_dir / "final_pool.json").exists() and (run_dir / "train_metrics.json").exists():
                return json.loads(result_path.read_text(encoding="utf-8"))
            return _original_run_search(config_path, method, reward, seed, steps, experiment_id, resume)

    _search_run_module.run_search = _locked_run_search
