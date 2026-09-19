# Corrections

Conclusions that were reported and later found wrong. Kept as a standing record
rather than folded silently into the current write-up, because several of them
were stated confidently and acted on.

## The root cause

`ToyActivations` seeded its own `torch.Generator` with the run seed, while
callers seeded model initialisation with `torch.manual_seed(<same seed>)`. Same
generator algorithm, same seed, same stream — so `nn.init.normal_` on a decoder
drew the same numbers as the concept matrix, and the first rows of `W_dec_a`
were **exactly the ground-truth concepts**. An untrained model scored
`best_cos = 1.0000` against ground truth.

Decoders initialised to real, frequent concepts are immediately useful for
reconstruction, so they stayed alive and carried activation mass. That inflated
every exclusive-mass measurement, and invalidated concept matching outright.

Fixed by `_DATA_SEED_OFFSET` in `diffdiff/data/toy.py`, with a regression test
asserting no decoder starts out parallel to a concept.

**How it was caught:** a metric pinned to an implausibly round constant across
seeds — leakage at exactly 0.5025 on all three DFC seeds, and untrained
`leak_a` exactly 1.0000 with `leak_b` exactly 0.0000. Learned quantities do not
do that. An earlier round of the same signal had already exposed a *different*
real bug (matching against concepts invisible to the scored model); the constant
persisted afterwards, which is what prompted looking again instead of writing up.

## What changed

| Claim as reported | Corrected | Direction |
|---|---|---|
| Overcomplete null floor 0.0740, FAIL on all seeds | **0.0012, PASS on all seeds** | 60× lower |
| Undercomplete null and control ranges overlap (1.27×) | **Clean gap, 1.69×** | discriminates |
| Detection limit 3.1%–6.3% of concepts | **1.6%–3.1%** | ~2× lower |
| Falloff is "a cliff between adjacent points" | **Graded: 1.69, 1.38, 1.19, 1.07, 1.04** | not a cliff |
| Mirror pairs "present but graded" | **Absent** — excess 0.000 on every seed | unsupported |
| Target frequency "was not carrying the result" | **It was** — mid-frequency n=16 fails to clear | reversed twice |
| Standard crosscoder beats DFC 3× recall, 9× precision (overcomplete) | **1.6× and 2×**, and DFC leaks 37× less | much smaller |

## Two retractions of retractions

Worth flagging separately, because in both cases a correction was itself wrong.

**Capacity regime.** `DESIGN.md` §6.1 originally predicted that an overcomplete
dictionary would leave the exclusive partitions dead. Contaminated data appeared
to show the opposite, so the section was rewritten to claim overcomplete gives
*twice* the floor, with a mechanism in which scarcity suppresses the artifact.
The corrected floors are undercomplete 0.0315 against overcomplete 0.0012 — a
25× gap the original way round. The first prediction was right; the correction
was the error, and has been reverted with a history note.

**Target frequency.** The sweep's headline points all destroy the *most active*
concepts, which was flagged at the time as making the limit optimistic.
Contaminated data showed mid-frequency scoring nearly as well (1.46× vs 1.53×),
and that caveat was retracted as "not materialising", calling the limit "a
robust property of the method". Corrected, mid-frequency at n=16 does not clear
the null at all (1.29×) while high-frequency clears comfortably (1.69×). The
original caveat was right.

## What survived unchanged

- **`standard_extreme_frac = 0.0`** on every null seed, in both regimes. The
  standard crosscoder reports nothing exclusive on identical inputs while the
  DFC populates its partition. This was flagged as suspect (the contamination
  hit `W_dec_a` but not `W_dec_b`, and relative decoder norm is a ratio) and
  turned out unaffected.
- **The DFC's exclusive partitions are never empty when undercomplete** — floor
  0.0315, `exclusive_live_frac` 1.0 on every seed. The floor moved ~13%, not
  enough to change the finding.
- **The paper's central claim holds in the regime it states.** The DFC beats the
  standard crosscoder undercomplete on recall, precision and leakage.

## Process notes

Three separate incidents where a pattern match on command text matched the
shell containing the pattern: a `pkill -f` that killed the running experiments,
a `pgrep -f` guard that hung a sweep for ~16 minutes, and a diagnostic that
reported a sweep as running when it was matching itself. Any `pgrep -f` /
`pkill -f` on a string that also appears in the invoking command self-matches by
construction.

One incident of editing a shell script while it was executing — the exact hazard
that had been identified earlier when deferring a `git rebase` for the same
reason. Bash reads scripts incrementally from a file offset, so an edit can make
it resume mid-line. Long runs are now launched from a copy outside the
repository.
