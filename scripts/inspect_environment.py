from __future__ import annotations

import json
from pathlib import Path

from rlalpha.config import load_paths
from rlalpha.doctor import run_doctor


if __name__ == "__main__":
    config = Path(__file__).resolve().parents[1] / "configs/experiment/preliminary_screen.yaml"
    print(json.dumps(run_doctor(load_paths(config)), indent=2, default=str))

