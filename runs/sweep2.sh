#!/bin/bash
# Sensitivity sweep, re-run after the RNG collision fix.
# Now UNDERCOMPLETE: the corrected data shows that is the regime which
# discriminates consistently (1.69x, clean gap) while overcomplete has larger
# but erratic separation whose seeds overlap the null.
# Baseline: corrected null_undercomplete = 0.0315, range [0.0263, 0.0359].
set -u
cd /home/user/diff-the-diff
export PYTHONUNBUFFERED=1
# NOTE: an earlier version waited here on `pgrep -f "diffdiff.cli compare"`.
# That pattern matched a shell whose own command line contained the pattern
# text, so the guard never cleared and the sweep hung indefinitely. The
# cross-architecture runs are sequenced ahead of this script by the caller
# instead.
run () { local n=$1; shift; echo "### START $n $(date +%T)"
  python3 -m diffdiff.cli control "$@" --json "runs/${n}.json" 2>&1 | sed "s/^/[$n] /"
  echo "### DONE $n $(date +%T)"; }
C="--seeds 0 1 2 --d 128 --concepts 256 --k 16 --steps 1500 --batch-size 512 --lr 1e-3 \
   --eval-batches 10 --quiet --regime dropped --no-standard --features 128 --exclusive-frac 0.05 --noise 0.02"
for n in 8 4 2 1; do run "sweep2_n${n}" $C --n-excl "$n" --drop-offset 0; done
run sweep2_n16_mid $C --n-excl 16 --drop-offset 120
echo "### ALL DONE $(date +%T)"
