#!/bin/bash
# Sensitivity sweep: hold the dictionary overcomplete, vary how many concepts
# are destroyed, and find where the control stops separating from the null.
# Baseline is null_overcomplete: mean 0.0740, range [0.0649, 0.0798].
# n_excl=16 at offset 0 already exists as control_dropped_overcomplete (0.1129).
set -u
cd /home/user/diff-the-diff
export PYTHONUNBUFFERED=1
run () {
  local name=$1; shift
  echo "### START $name $(date +%T)"
  python3 -m diffdiff.cli control "$@" --json "runs/${name}.json" 2>&1 | sed "s/^/[$name] /"
  echo "### DONE $name $(date +%T)"
}
COMMON="--seeds 0 1 2 --d 128 --concepts 256 --k 16 --steps 1500 --batch-size 512 \
        --lr 1e-3 --eval-batches 10 --quiet --regime dropped --no-standard \
        --features 4096 --exclusive-frac 0.05 --noise 0.02"
for n in 8 4 2 1; do
  run "sweep_n${n}" $COMMON --n-excl "$n" --drop-offset 0
done
# Same count as the existing n=16 run, but destroying mid-frequency concepts
# instead of the most active ones, to see how much the optimistic choice matters.
run sweep_n16_mid $COMMON --n-excl 16 --drop-offset 120
echo "### ALL DONE $(date +%T)"
