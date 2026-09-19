#!/bin/bash
# Cross-architecture comparison: DFC vs standard crosscoder, scored against
# ground-truth concepts. d_a != d_b makes this genuinely cross-architecture.
set -u
cd /home/user/diff-the-diff
export PYTHONUNBUFFERED=1
echo "### START xarch_compare $(date +%T)"
python3 -m diffdiff.cli compare --seeds 0 1 2 \
  --d-a 128 --d-b 96 --concepts 256 --n-excl 16 \
  --features 4096 --exclusive-frac 0.05 --k 16 \
  --steps 1500 --batch-size 512 --lr 1e-3 --quiet \
  --json runs/xarch_compare.json 2>&1 | sed 's/^/[xarch] /'
echo "### DONE xarch_compare $(date +%T)"
