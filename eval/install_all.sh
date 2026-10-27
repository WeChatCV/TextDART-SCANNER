#!/usr/bin/env bash
# Resolve evaluator and project dependencies together.
set -euo pipefail
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
python -m pip install -r "${SCRIPT_DIR}/framework.txt" -e "${SCRIPT_DIR}/.."
python -m pip check
