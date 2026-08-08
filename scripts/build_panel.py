from __future__ import annotations

import json

from rlalpha.data.panel import build_panel


if __name__ == "__main__":
    print(json.dumps(build_panel("/data/sunyuxiang/rl_alpha", "/data/sunyuxiang/rl_alpha/processed"), indent=2))

