# Phase 0 results: the null test

> **UNDER REVISION — do not cite §3, §4, §6 or §7 below.**
>
> Every number in this document was produced before the RNG collision fix
> (`_DATA_SEED_OFFSET` in `diffdiff/data/toy.py`), which had initialised
> decoders to the ground-truth concepts themselves. Corrected re-runs are in
> progress.
>
> Known so far: the overcomplete null floor falls from **0.0740 to 0.0012**
> (60x) and now PASSES, and the undercomplete floor from 0.0363 to 0.0315. The
> two regimes rank the opposite way round from what §3.2 claims.
>
> **The §8 recommendation to abandon quantization is therefore under review.**
> It rested on a 7.4% noise floor in the only regime that discriminated; the
> real floor there is 0.12%. The detection limit in §6 was computed against a
> floor roughly 60x too high and must be re-measured before any conclusion about
> quantization stands.

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


---

# Phase 0.5: the sensitivity sweep

## 5. What was run

Phase 0 established that the method discriminates *only* when the dictionary is
overcomplete, and only measured that at one damage level (16 of 256 concepts
destroyed, ~6%). That level is far cruder than real quantization damage, so the
open question was: **how small can the damage get before it becomes invisible?**

The sweep holds everything fixed — 4096 features, 3 seeds, identity transform,
2% noise — and varies only the number of concepts destroyed in model B. The
fixed baseline is `null_overcomplete`: **0.0740, range 0.0649 – 0.0798**.

A point counts as detectable only if its per-seed *minimum* clears the null's
*maximum* (0.0798). Comparing means alone would let an overlapping distribution
pass.

## 6. The sensitivity curve

| Destroyed | % of concepts | mean | range | vs null | Detectable |
|---|---|---|---|---|---|
| 16 (most active) | 6.3% | 0.1129 | 0.1020 – 0.1294 | **+52.6%** | **yes** |
| 16 (mid-frequency) | 6.3% | 0.1082 | 0.0959 – 0.1203 | **+46.3%** | **yes** |
| 8 | 3.1% | 0.0803 | 0.0674 – 0.0956 | +8.6% | no |
| 4 | 1.6% | 0.0689 | 0.0531 – 0.0770 | −6.8% | no |
| 2 | 0.8% | 0.0742 | 0.0634 – 0.0852 | +0.4% | no |
| 1 | 0.4% | 0.0768 | 0.0611 – 0.0910 | +3.8% | no |

**The detection limit sits between 3.1% and 6.3% of all concepts destroyed
outright**, and it is a cliff rather than a slope: halving the damage from 16 to
8 concepts collapsed the signal from +52.6% to +8.6% with full range overlap.

Below n=8 the readings are **flat noise**. They scatter around the null in both
directions (−6.8%, +0.4%, +3.8%) with no monotone relationship to the amount of
damage. At n=4 the control read *lower* than destroying nothing at all. The sign
of the difference carries no information at these levels.

### 6.1 The limit does not depend on which concepts are destroyed

Concept frequency in the toy model follows a power law in the concept index, so
destroying concepts starting at index 0 targets the **most active** concepts —
the easiest possible case. The concern was that this made the one detectable
point an artifact of generous target selection, and that the true limit was
worse than 6.3%.

`sweep_n16_mid` destroys the same number of concepts at mid frequency (offset
120) and **separates almost as well**: +46.3% against +52.6%, clearing the null
ceiling by a comfortable margin (min 0.0959 vs 0.0798).

So the concern did not materialise. Target frequency was not carrying the
result, and the detection limit is a robust property of the method at ~6% of
concepts rather than an optimistic artifact. This makes the finding cleaner, and
leaves the conclusion in §7 unchanged: 6% of a model's concepts destroyed
outright is far more damage than quantization causes.

## 7. What this means for the quantization use case

Int4 quantization does not delete several percent of a model's concepts. It
perturbs every weight slightly and degrades some features partially. The damage
this use case exists to detect is **orders of magnitude below the measured
detection limit**.

This triggers a kill criterion, though not one of the three written down in
`DESIGN.md` §12. Those anticipated the method producing *false positives* —
§11's mirror-features concern. It does produce them (a 7.4% floor on identical
inputs), but the disqualifying problem is **sensitivity**: the method cannot see
damage of the size we care about, whatever its floor.

Two secondary findings reinforce this:

- **The standard crosscoder reports nothing exclusive** (`standard_extreme_frac
  = 0.0`) on identical inputs, where the DFC populates every exclusive slot. In
  the near-identical regime the architectural fix is worse than its baseline.
- **Discrimination requires an overcomplete dictionary**, which is the expensive
  axis on 8GB and the one that drives the auto-interp bill, since the paper's
  ~500,000 Claude queries per experiment scale with dictionary size.

## 8. Recommendation

**Stop work on quantization and merge verification. Pivot to the
cross-architecture use case.**

The evidence for the pivot target is already in hand: our own cross-architecture
control separated at **3.16×** (0.2335 vs the 0.0740 null) with a wide
non-overlapping gap — the single strongest separation anywhere in these runs.
That matches the paper's own findings, which are all cross-architecture
(Llama-vs-Qwen, GPT-OSS-vs-DeepSeek), and it is the setting the method was built
for. Differences between independently trained models are large; differences
introduced by rounding are not.

Concretely, that means **vendor-migration risk** and **open-weight adoption
diligence** (`README.md` options B and A), both of which need weights for two
models — which an enterprise evaluating open-weight models has.

**What carries over, and it is most of the work.** The harness is regime-agnostic:
the crosscoder and DFC, the training loop, the memmap cache, the toy generator,
the mirror detector, and the CLI all apply unchanged. `align/` already contains
the paper's Algorithm 1 window-expansion aligner for the cross-tokenizer case,
tested against the paper's own `['1989']` example. The pivot is a change of
target, not a rewrite.

**What we should publish either way.** The null test and the sensitivity curve
are a genuine contribution. The paper reports no null test, so it has no measured
false-discovery floor and no sensitivity curve — which means there is currently
no published answer to "how large must a difference be before this method can see
it?" Our answer, at toy scale, is ~5% of concepts destroyed outright. That is
decision-useful for anyone considering model diffing for a near-identical
comparison, and it is cheap to reproduce.

**Revised priority order:**

1. **Re-run the null and sensitivity sweep on real models** (`docs/HANDOFF.md`
   step 2), to check whether the toy geometry transfers. This is the main risk
   to the conclusion above.
2. **The LLM black-box baseline** (Phase 1), unchanged in importance. It remains
   the control that decides whether white-box access is worth its cost, and the
   paper does not compare against it.
3. **Cross-architecture diffing** as the new primary target, with the null test
   retained as a standing check rather than a one-off.
