# Results

Corrected data only. Superseded numbers are in
[`PHASE0-RESULTS-SUPERSEDED.md`](PHASE0-RESULTS-SUPERSEDED.md); what changed and
why is in [`CORRECTIONS.md`](CORRECTIONS.md).

All runs use the synthetic concept model (`diffdiff/data/toy.py`), 1500 steps,
batch 512. Scripts in `runs/*.sh`, raw output in `runs/*.json`.

## 1. The null test

Diff a model against itself. Inputs are bit-identical (asserted), so every
feature landing in an exclusive partition is a false positive by construction.
The source paper reports no such test.

| Regime | Dictionary | exclusive mass | range | live features | verdict |
|---|---|---|---|---|---|
| Undercomplete | 128 | **0.0315** | 0.0263 – 0.0359 | **100%** | MARGINAL |
| Overcomplete | 4096 | **0.0012** | 0.0001 – 0.0030 | 1–4% | PASS |

**Capacity governs the false-discovery floor, a 25× gap.** When the dictionary
is overcomplete the shared partition alone explains the data, so the optimizer
leaves the exclusive partitions dead. When undercomplete, shared capacity is
scarce, exclusive features get recruited, and the floor rises.

**The standard crosscoder reports nothing exclusive in either regime**
(`standard_extreme_frac = 0.0`, all seeds). On identical inputs a DFC populates
every one of its exclusive slots while its own baseline correctly finds nothing.
That comparison is the sharpest single result here, and it was never affected by
the contamination.

**No mirror pairs.** Mirror excess is 0.000 on every null seed. The
`DESIGN.md` §11 prediction that partitions would fill with mirrored duplicates
is unsupported; the mechanism is plain forced allocation.

## 2. Sensitivity: how small can a difference get?

Undercomplete (the regime that discriminates consistently), against the 10-seed
null: mean 0.0344, sd 0.0048, max 0.0431.

**Per-run detection rate is the decision-relevant metric.** An audit runs one
diff, not ten, so what matters is how often a *single* run clears the null —
not whether the means differ significantly. These diverge sharply: at n=4 the
means are statistically distinguishable (z = 2.8) while only 3 runs in 10 would
actually detect anything.

| Damage | % of concepts | mean | sep | z | **runs detecting** |
|---|---|---|---|---|---|
| 32 concepts, mid-frequency | 12.5% | 0.0511 | 1.48× | 6.2 | **9/10** |
| 8 concepts, most-active | 3.1% | 0.0466 | 1.36× | 5.3 | **7/10** |
| 16 concepts, mid-frequency | 6.3% | 0.0438 | 1.27× | 4.4 | **7/10** |
| 4 concepts, most-active | 1.6% | 0.0407 | 1.18× | 2.8 | **3/10** |

Three-seed curve (same direction, coarser): 1.69× at n=16, 1.38× at n=8, 1.19×
at n=4, 1.07× at n=2, 1.04× at n=1 — a graded decay toward 1.0, not a cliff.

**Which concepts are destroyed matters, at roughly a 2× penalty.** Mid-frequency
damage at n=16 is about as detectable as most-active damage at n=8. Concept
frequency follows a power law, so destroying a model's most-used concepts is the
easy case; compression damage has no reason to land there.

**Practical limit: ~3% of concepts destroyed outright for ~70% per-run
detection**, or ~6% when the damage is not concentrated on the most-used
concepts.

## 3. Cross-architecture: DFC vs standard crosscoder

Genuinely different architectures (`d_a=128`, `d_b=96`), 16 planted exclusive
concepts per model, scored against ground truth with a **matched exclusive
budget** — a DFC's dedicated partition against the same number of most-extreme
relative-decoder-norm features. Without matching, an architecture that labels
more features exclusive scores higher on recall for free.

| Regime | | recall | precision (of ceiling) | shared leakage |
|---|---|---|---|---|
| Undercomplete | **DFC** | **0.062** | **0.167** | **0.056** |
| | standard | 0.021 | 0.056 | 0.333 |
| Overcomplete | DFC | 0.219 | 0.198 | **0.005** |
| | **standard** | **0.344** | **0.385** | 0.188 |

**The paper's central claim holds in the regime it states.** Undercomplete — the
regime the paper identifies as where real applications operate — the DFC wins on
all three measures.

**The ranking crosses over with capacity.** Overcomplete, post-hoc selection
wins on recall and precision: a standard crosscoder picks its most-extreme
features *after* training from the whole dictionary, while a DFC commits its
slots at initialisation and the optimizer fills them regardless.

**But the DFC leaks shared concepts 6–37× less in both regimes.** It is
conservative: 98% of its exclusive features match nothing rather than
confidently mislabelling shared structure. For an audit tool a leaked shared
concept is a false difference someone investigates and finds spurious, so on
that axis the DFC wins even where it loses on recall.

Absolute numbers are low throughout — a recall of 0.062 is about 1 concept in
16. Neither architecture is doing well at this scale.

## 4. Recommendation

**Stay pivoted to cross-architecture. Do not return to quantization.**

The reasoning differs from the one originally given, and most of that original
reasoning was wrong (see `CORRECTIONS.md`). Three of the four original arguments
collapsed: the noise floor was 60× overstated, the "cliff" was a graded falloff,
and the mirror-pair evidence evaporated. What survives:

1. **Scale mismatch.** Reliable per-run detection needs ~3% of a model's
   concepts destroyed *outright*, and ~6% when damage is not concentrated on the
   most-used concepts. Quantization perturbs weights; it does not delete several
   percent of concepts. The gap remains orders of magnitude.
2. **Cross-architecture is where the signal is.** The same harness separates
   far more strongly on genuinely different models, which is also where the
   paper's own findings come from.

**Retained as the contribution regardless of use case:** the null test, the
false-discovery floor, and the per-run sensitivity curve. The paper reports no
null test and therefore has no measured floor and no answer to "how large must a
difference be before this method can see it?"

## 5. Caveats

- **Toy scale.** Everything here is a synthetic concept model, not a real
  network. Whether the geometry transfers is the single largest risk to every
  conclusion above, and it needs a GPU and model weights —
  see [`HANDOFF.md`](HANDOFF.md).
- **Seed counts.** Section 2 uses 10 seeds; sections 1 and 3 use 3. Three seeds
  proved insufficient to resolve differences smaller than the null's own spread
  and produced two conclusions that 10 seeds overturned. Section 3 should be
  re-run at 10 seeds before its crossover is relied on.
- **One dictionary pair per regime.** Capacity turned out to be the governing
  variable in both phases, and it was sampled at only two points (128 and 4096).
