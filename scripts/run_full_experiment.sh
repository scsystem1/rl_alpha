#!/usr/bin/env bash
set -euo pipefail

echo "Formal full experiment is blocked: a fresh one-prompt/eight-answer GPU smoke," >&2
echo "real interrupted-vs-uninterrupted equivalence, the predeclared 12-cell small" >&2
echo "matrix, and clean-clone reproduction are incomplete. Use" >&2
echo "configs/experiment/revision_v3_cpu_smoke.yaml only for" >&2
echo "the documented Random/GP engineering smoke; see docs/revision_compliance_matrix.md." >&2
exit 2
