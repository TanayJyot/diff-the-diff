#!/bin/bash
# Clean control: identity transform (no misalignment confound) with concepts
# destroyed in B only. This is the direct quantization analogue.
set -u
cd /home/user/diff-the-diff
export PYTHONUNBUFFERED=1
run () {
  local name=$1; shift
  echo "### START $name $(date +%T)"
  python3 -m diffdiff.cli "$@" --json "runs/${name}.json" 2>&1 | sed "s/^/[$name] /"
  echo "### DONE $name $(date +%T)"
}
COMMON="--seeds 0 1 2 --d 128 --concepts 256 --k 16 --steps 1500 --batch-size 512 --lr 1e-3 --eval-batches 10 --quiet --regime dropped --no-standard"
run control_dropped_undercomplete control $COMMON --features 128  --exclusive-frac 0.05 --n-excl 16 --noise 0.02
run control_dropped_overcomplete  control $COMMON --features 4096 --exclusive-frac 0.05 --n-excl 16 --noise 0.02
echo "### ALL DONE $(date +%T)"
