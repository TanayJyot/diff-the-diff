#!/bin/bash
# Resolve the detection boundary with 10 seeds instead of 3.
# The 3-seed sweep puts the limit between n=4 and n=8, but the margins
# (n=8 clears by +0.0031, n=4 misses by -0.0032) are thinner than the null's
# own spread (0.0096), so three seeds cannot separate them.
set -u
cd /home/user/diff-the-diff
export PYTHONUNBUFFERED=1
SEEDS="0 1 2 3 4 5 6 7 8 9"
run () { local n=$1; shift; echo "### START $n $(date +%T)"
  python3 -m diffdiff.cli "$@" --json "runs/${n}.json" 2>&1 | sed "s/^/[$n] /"
  echo "### DONE $n $(date +%T)"; }
C="--seeds $SEEDS --d 128 --concepts 256 --k 16 --steps 1500 --batch-size 512 --lr 1e-3 \
   --eval-batches 10 --quiet --features 128 --exclusive-frac 0.05 --no-standard"
run boundary_null null    $C
run boundary_n8   control $C --regime dropped --n-excl 8 --noise 0.02 --drop-offset 0
run boundary_n4   control $C --regime dropped --n-excl 4 --noise 0.02 --drop-offset 0
echo "### ALL DONE $(date +%T)"
