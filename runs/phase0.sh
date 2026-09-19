#!/bin/bash
# Phase 0 experiment sweep. Null and positive control, in both capacity regimes.
set -u
cd /home/user/diff-the-diff
export PYTHONUNBUFFERED=1
run () {
  local name=$1; shift
  echo "### START $name $(date +%T)"
  python3 -m diffdiff.cli "$@" --json "runs/${name}.json" 2>&1 | sed "s/^/[$name] /"
  echo "### DONE $name $(date +%T)"
}
COMMON="--seeds 0 1 2 --d 128 --concepts 256 --k 16 --steps 1500 --batch-size 512 --lr 1e-3 --eval-batches 10 --quiet"
run null_undercomplete  null    $COMMON --features 128  --exclusive-frac 0.05
run control_undercomplete control $COMMON --features 128  --exclusive-frac 0.05 --regime cross-arch --n-excl 16 --no-standard
run null_overcomplete   null    $COMMON --features 4096 --exclusive-frac 0.05
run control_overcomplete control $COMMON --features 4096 --exclusive-frac 0.05 --regime cross-arch --n-excl 16 --no-standard
echo "### ALL DONE $(date +%T)"
