from __future__ import annotations

import argparse
import json

from rlalpha.config import load_paths
from rlalpha.data.panel import build_panel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build a fully identified panel from an explicit project config.")
    parser.add_argument("--config", default="configs/experiment/revision_v3_cpu_smoke.yaml")
    arguments = parser.parse_args()
    paths = load_paths(arguments.config)
    print(json.dumps(build_panel(paths.raw_data_root, paths.processed_root), indent=2))
