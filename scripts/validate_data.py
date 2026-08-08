from __future__ import annotations

import json

from rlalpha.data.validate import validate_raw_bundle


if __name__ == "__main__":
    report = validate_raw_bundle("/data/sunyuxiang/rl_alpha")
    print(json.dumps(report, indent=2, default=str))
    raise SystemExit(0 if report["ok"] else 2)

