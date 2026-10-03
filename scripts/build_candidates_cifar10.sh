#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
python build_candidates.py --dataset cifar10 --config configs/default.json "$@"