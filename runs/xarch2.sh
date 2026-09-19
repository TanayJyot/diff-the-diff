#!/bin/bash
# Cross-architecture comparison in BOTH capacity regimes. The paper claims the
# DFC's advantage is largest when undercomplete, so that regime is the one that
# actually tests its central claim.
set -u
cd /home/user/diff-the-diff
export PYTHONUNBUFFERED=1
run () {
  local name=$1; shift
  echo "### START $name $(date +%T)"
  python3 -m diffdiff.cli compare "$@" --json "runs/${name}.json" 2>&1 | sed "s/^/[$name] /"
  echo "### DONE $name $(date +%T)"
}
COMMON="--seeds 0 1 2 --d-a 128 --d-b 96 --concepts 256 --n-excl 16 \
        --exclusive-frac 0.05 --k 16 --steps 1500 --batch-size 512 --lr 1e-3 --quiet"
run xarch_undercomplete $COMMON --features 128
run xarch_overcomplete  $COMMON --features 4096
echo "### ALL DONE $(date +%T)"
