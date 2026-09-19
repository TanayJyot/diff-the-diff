# Phase 0 results: the null test

> **Pre-registration.** The reading criteria in §2 were written and committed
> *before* the numbers came back, so that a result cannot be rationalised after
> the fact. Numbers land in §3.

## 1. What was run

Diff a model against itself and count what the method invents. The toy source
(`diffdiff/data/toy.py`) produces `x_a` and `x_b` that are **bit-identical** —
asserted, not assumed — so every feature landing in an exclusive partition is a
false positive by construction.

Four runs, three seeds each, on the synthetic concept model:

| Run | Dictionary | Concepts | Regime | Differences present? |
|---|---|---|---|---|
| `null_undercomplete` | 128 | 256 | undercomplete | none |
| `control_undercomplete` | 128 | 256 | undercomplete | 16 exclusive per model |
| `null_overcomplete` | 4096 | 256 | overcomplete | none |
| `control_overcomplete` | 4096 | 256 | overcomplete | 16 exclusive per model |

Both capacity regimes, because they have opposite incentives. Overcomplete: the
shared partition alone can explain everything, so the optimizer may leave the
exclusive partitions dead — a clean null that says more about spare capacity
than about the method. Undercomplete: shared capacity is scarce, there is real
pressure to recruit exclusive features, and this is the regime the paper says
real applications occupy.

Each null is paired with a positive control in the *same* regime, scored by the
same code. Without it a clean null is vacuous: if the partitions also stay empty
where differences genuinely exist, the method simply is not finding anything.

## 2. How to read it (fixed in advance)

**Headline metric — `exclusive_mass_frac`:** the share of total feature
activation mass routed through partitions that should be empty. A method that
correctly finds nothing drives this toward zero. The reference point for "no
preference" is the uniform share (here `2 × 0.05 = 0.10`), which is *not* the
same as correct.

**Second metric — `mirror_excess`:** how much more often an exclusive feature
has a near-duplicate in the opposing partition than shared features do by
chance. The control is two disjoint random subsets of the shared partition,
sized to match the real comparison, because the statistic is a maximum over
candidates and grows with pool size.

Verdicts, as encoded in `NullReport.verdict()`:

| Condition | Verdict |
|---|---|
| `exclusive_mass_frac < 0.01` and `mirror_excess < 0.05` | **PASS** — partitions stayed essentially empty |
| `mirror_excess >= 0.20` | **FAIL** — partitions filled with manufactured differences |
| `exclusive_mass_frac >= 0.05` | **FAIL** — substantial mass through empty-by-construction partitions |
| otherwise | **MARGINAL** — inspect features individually |

**What each joint outcome would mean:**

- *Null PASS + control finds differences* → the near-identical regime is usable.
  Quantization stays the use case; proceed to Phase 1.
- *Null FAIL (mirroring) + control finds differences* → the detector works and
  the DFC manufactures differences when there are none. This is DESIGN §11
  confirmed, and it makes the mirror problem the research target rather than a
  blocker.
- *Null PASS + control finds nothing* → **vacuous**. The method is not
  discriminating at this scale; the null tells us nothing and the toy model or
  the training budget needs work before any conclusion.
- *Null FAIL + control finds nothing* → the harness is broken, not the method.

**A result that holds only in one capacity regime is a result about that
regime**, and the undercomplete one is the one that transfers to real use.

## 3. Results

Six runs, three seeds each. `exclusive_mass_frac` is the share of feature
activation mass routed through the exclusive partitions.

| Run | Dictionary | mean | per-seed range | vs paired null |
|---|---|---|---|---|
| `null_undercomplete` | 128 | 0.0363 | 0.0296 – 0.0405 | — |
| `control_undercomplete` (cross-arch) | 128 | 0.0632 | 0.0523 – 0.0797 | 1.74×, no overlap |
| `control_dropped_undercomplete` (clean) | 128 | 0.0461 | 0.0381 – 0.0524 | **1.27×, OVERLAPS** |
| `null_overcomplete` | 4096 | 0.0740 | 0.0649 – 0.0798 | — |
| `control_overcomplete` (cross-arch) | 4096 | 0.2335 | 0.2069 – 0.2851 | 3.16×, no overlap |
| `control_dropped_overcomplete` (clean) | 4096 | 0.1129 | 0.1020 – 0.1294 | **1.53×, no overlap** |

The uniform share is 0.094 (undercomplete) and 0.100 (overcomplete).

### 3.1 The exclusive partitions are never empty

Given **bit-identical inputs**, the DFC routed 3.6% (undercomplete) and 7.4%
(overcomplete) of activation mass through partitions where every feature is a
false positive by construction. `exclusive_live_frac` was 1.0 in every
undercomplete run — all twelve exclusive features alive.

For contrast, the standard crosscoder on the same data reported
`standard_extreme_frac = 0.0` across all seeds: **no** feature reached an
extreme relative decoder norm. In the near-identical regime the architectural
fix is strictly worse than the baseline it improves on, because its capacity is
allocated up front rather than earned.

Neither result appears in the paper, which reports no null test.

### 3.2 Capacity moves the floor the opposite way to our prediction

Overcomplete has roughly **twice** the false-discovery floor of undercomplete
(0.074 vs 0.036), landing near the uniform share. The prediction recorded in
`DESIGN.md` §6.1 was the reverse, and the mechanism runs the other way: when
capacity is scarce a shared feature is a better bargain, since it reconstructs
*both* models per slot spent, so the optimizer economises by avoiding the
exclusive partitions. Abundant capacity removes that pressure.

**Dictionary size is therefore a false-discovery knob, not merely a capacity
choice.**

### 3.3 The cross-architecture control was confounded, and it flattered the method

The cross-arch control differs from the null in *two* ways: exclusive concepts
exist **and** a random affine transform misaligns the two representational
spaces. Its separation therefore conflates finding real differences with
struggling to align different spaces.

Removing the confound cut separation from 1.74× to 1.27× (undercomplete) and
from 3.16× to 1.53× (overcomplete). The confound was supplying roughly half the
apparent signal. Quantization involves no misalignment at all, so the clean
control is the one that speaks to our use case.

### 3.4 Discrimination survives only when overcomplete

With the confound removed:

- **Undercomplete: the null and control ranges overlap.** Control seed 1
  (0.0381) falls inside the null's range (0.0296–0.0405). A run with 16
  genuinely destroyed concepts read the same as a run where nothing differed.
- **Overcomplete: a clean gap.** Max null 0.0798 against min control 0.1020.

This directly tensions the paper's framing. It reports DFC's advantage as
largest in the *undercomplete* regime — "most likely the regime in which
real-world applications operate" — but measures **recall of exclusive concepts**
on a toy model, with no null baseline. A method that populates its exclusive
partition more readily scores better on recall while getting worse at telling
signal from nothing. Recall without a null is not discrimination.

### 3.5 Mirroring is present but is not the main mechanism

`mirror_excess` stayed small throughout (0.00–0.06 undercomplete, 0.01–0.05
overcomplete). Where pairs did cross threshold, decoder cosine was 0.87 —
unambiguous duplicates — and median best-partner correlation ran 1.5–5.8× the
control on every seed, so mirroring is real but graded, sitting mostly below a
0.5 cutoff.

The dominant mechanism is simpler than mirroring: **forced allocation**. The
partition exists, so it gets filled.

## 4. What this decides

## 4. What this decides

**The regime is usable, but only in a configuration we cannot currently
afford, and with a signal margin too thin to trust yet.**

Against the pre-registered criteria: both nulls fail or are marginal, and the
clean control separates only when overcomplete. That is the "null FAIL + control
finds differences" branch — the detector works, and the DFC manufactures
differences when there are none.

The constraint this imposes is concrete. Discrimination required a dictionary
15× overcomplete relative to the concepts in the data (4096 for 272). Real
models carry far more concepts than a toy, so matching that ratio at real scale
means a very large dictionary — which is the expensive axis on 8GB, and which
also drives the auto-interp bill, since the paper's ~500,000 Claude queries per
experiment scale with dictionary size.

And even there the margin is thin: 1.53× separation was obtained by destroying
**16 of 272 concepts outright (~6%)**. Real int4 quantization damage is far
subtler than deleting 6% of a model's concepts.

**Recommended next step, before any further investment: a sensitivity sweep.**
Hold the dictionary overcomplete and vary the number of destroyed concepts
(16 → 8 → 4 → 2 → 1), finding where the null and control distributions start to
overlap. That yields a *detection limit in concepts-destroyed units*, which is
the single number that says whether realistic quantization damage is detectable
at all. It is cheap, it is decisive, and it should come before Phase 1.

If the detection limit lands well above realistic quantization damage, the
honest move is to pivot to the cross-architecture use case, where differences
are large and the method demonstrably works — and to report the detection limit
as the finding, since the paper reports no null test and therefore has no
measured floor to compare against.
