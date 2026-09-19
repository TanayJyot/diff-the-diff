#!/bin/bash
# Re-run everything invalidated by the RNG collision fix.
set -u
cd /home/user/diff-the-diff
export PYTHONUNBUFFERED=1
run () { local n=$1; shift; echo "### START $n $(date +%T)"
  python3 -m diffdiff.cli "$@" --json "runs/${n}.json" 2>&1 | sed "s/^/[$n] /"
  echo "### DONE $n $(date +%T)"; }
S="--seeds 0 1 2 --d 128 --concepts 256 --k 16 --steps 1500 --batch-size 512 --lr 1e-3 --eval-batches 10 --quiet"
X="--seeds 0 1 2 --d-a 128 --d-b 96 --concepts 256 --n-excl 16 --exclusive-frac 0.05 --k 16 --steps 1500 --batch-size 512 --lr 1e-3 --quiet"
run null_undercomplete  null $S --features 128  --exclusive-frac 0.05
run null_overcomplete   null $S --features 4096 --exclusive-frac 0.05
run control_dropped_undercomplete control $S --features 128  --exclusive-frac 0.05 --regime dropped --n-excl 16 --noise 0.02 --no-standard
run control_dropped_overcomplete  control $S --features 4096 --exclusive-frac 0.05 --regime dropped --n-excl 16 --noise 0.02 --no-standard
run xarch_undercomplete compare $X --features 128
run xarch_overcomplete  compare $X --features 4096
echo "### ALL DONE $(date +%T)"
