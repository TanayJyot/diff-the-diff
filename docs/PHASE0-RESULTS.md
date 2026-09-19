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

_Pending — runs in progress._

## 4. What this decides

Per `DESIGN.md` §11 and §12, Phase 0 decides whether the project continues on
quantization/merge verification or pivots to the cross-architecture use case,
where the method demonstrably works. That decision is recorded here once §3 is
filled in.
