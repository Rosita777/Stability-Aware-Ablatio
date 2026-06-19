#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
python3 src/reproduce_main_results.py --data-dir data/main --output-dir outputs
