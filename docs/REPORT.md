# diff-the-diff — where the project stands

*Status report, 2026-09-20. ~10 minute read.*

---

## The one-paragraph version

We set out to build an industry product around Anthropic's cross-architecture
model-diffing paper. We picked **quantization verification** as the first use
case, built a test harness, and ran the one experiment the paper never did — a
null test. The results killed that use case, so we pivoted to
**cross-architecture diffing**. Midway through the pivot I found a bug that had
contaminated most of the earlier numbers, re-ran everything, and several
conclusions reversed. The final recommendation is unchanged — stay pivoted —
but for different reasons than the ones I originally gave.

---

## 1. What the paper actually is

Not code diffs. It's mechanistic interpretability.

**Model diffing** compares the *internals* of two neural networks to find what
one has learned that the other hasn't — without being told what to look for.
The point is catching "unknown unknowns": behaviours no evaluation suite was
written to test. The paper's own example is that this could have flagged the
sycophancy in an April GPT-4o update before release.

The tool is a **crosscoder**: a shared dictionary of concepts learned jointly
across two models, sorting each concept into *only in A*, *only in B*, or *in
both*. The paper's contribution is the **DFC** (Dedicated Feature Crosscoder),
which reserves a fixed slice of the dictionary — say 5% — for each model's
exclusive concepts, rather than deciding exclusivity after training.

That reservation is the crux of everything below. **Those slots exist whether or
not there is anything real to put in them.**

---

## 2. Why we chose quantization first

Quantization shrinks a model to run cheaper. Standard practice checks perplexity
and a benchmark sweep, which catches *average* degradation but is nearly blind
to *selective* damage — the model is 99% as good and 1% worse exactly where it
matters.

We picked it because it's the one setting with something close to **ground
truth**: you know what the transformation was meant to preserve. That makes the
method falsifiable rather than merely demonstrable, which the published work
largely isn't.

Known risk going in: the paper admits the method struggles on very similar
models, and a quantized model is *extremely* similar to its parent.

---

## 3. The experiment that decided it

**The null test: diff a model against itself.**

If the two inputs are identical, the right answer is *nothing*. So anything the
tool reports as a difference is definitely wrong, and counting those measures
how much it fabricates. **The paper never runs this.**

We also ran the opposite as a control — two models that genuinely *do* differ —
because "it found nothing" is meaningless if the tool is simply broken.

### What we found

| | Result |
|---|---|
| Identical inputs, cramped dictionary | 3.2% of activity reported as "differences" — all fabricated |
| Identical inputs, roomy dictionary | 0.1% — essentially clean |
| **Standard crosscoder, identical inputs** | **Reports nothing, correctly** |

That last row is the sharpest result we have. On identical inputs the *standard*
crosscoder finds nothing — correct — while the DFC fills every reserved slot.
The architectural "improvement" is worse than its own baseline in this regime,
because its capacity is allocated up front rather than earned.

We then measured **how small a difference can get before it becomes invisible**.
Running 10 seeds per point:

- ~3% of a model's concepts destroyed outright → detected in **7 runs out of 10**
- ~1.6% destroyed → detected in **3 out of 10**
- If the damage isn't concentrated on the model's most-used concepts, roughly
  **double** the damage is needed for the same reliability

**Verdict for quantization:** int4 quantization perturbs weights; it doesn't
delete 3–6% of a model's concepts. The damage we'd be hunting is orders of
magnitude below what the method can see. Use case closed.

---

## 4. The pivot

**To cross-architecture diffing** — comparing two genuinely different models
(vendor migration, open-weight adoption diligence). Differences between
independently trained models are large; differences from rounding are not. This
is also where all the paper's own findings come from.

Most of the work carried over — the harness is regime-agnostic.

For the pivot I built a **ground-truth scorer**: since our synthetic model knows
which concepts are genuinely exclusive, we can ask directly whether the method
found *those*, or just filled its partition. Crucially it uses a **matched
budget**, so neither architecture wins by simply labelling more things
exclusive.

**Result:** the paper's central claim holds where it makes it. In the cramped
(undercomplete) regime — which the paper says is where real applications live —
the DFC beats the standard crosscoder on every measure. In the roomy regime the
ranking flips, because picking your best features *after* training beats
committing slots in advance. But the DFC mislabels shared concepts as
differences **6–37× less often** in both regimes, which matters a lot for an
audit tool where every false difference is someone's wasted investigation.

---

## 5. The bug, and what it cost

Partway through the pivot, a metric came back pinned to exactly the same value
across three different random seeds. Learned quantities don't do that.

**The cause:** the synthetic data generator and the model initialisation were
seeded with the same number, using the same random algorithm. They drew *the
same random numbers*. The model's starting weights were literally the
ground-truth concepts it was supposed to discover.

This inflated most measurements. After fixing it and re-running everything:

| Claim I had reported | Corrected |
|---|---|
| Noise floor 7.4%, failing | **0.1%, passing** (60× error) |
| Detection limit 3–6% of concepts | **1.6–3%** |
| Signal collapses off "a cliff" | **Graded decline** |
| Mirror-duplicate features present | **Absent entirely** |
| Which concepts are destroyed doesn't matter | **It does — ~2× penalty** |

Seven reported conclusions were wrong. Two of them were cases where I had
"corrected" a *correct* original prediction on the strength of bad data — the
retraction was the error, not the original claim.

I also asserted things from 3 seeds that 10 seeds later overturned, twice. The
honest lesson is that three seeds could not resolve differences smaller than the
data's own noise, and I should have run more before making claims either way.

All of this is recorded in `docs/CORRECTIONS.md` rather than quietly fixed, and
the superseded results file is kept rather than overwritten.

---

## 6. Where it stands now

**The recommendation is unchanged — stay on cross-architecture — but it now
rests on one argument instead of four.** Three of the original four collapsed
with the bug. What survives is the scale mismatch: reliable detection needs
several percent of a model's concepts *deleted*, and quantization doesn't do
that.

**What's genuinely worth keeping, regardless of use case:**

1. **The null test and the measured false-discovery floor.** No published work
   runs this, so there is currently no answer to "how large must a difference be
   before this method can see it?" We have one.
2. **Capacity is the governing variable** — dictionary size sets the noise floor
   (25× swing) *and* decides which architecture wins. The paper reports neither.
3. **The standard-crosscoder comparison** — it correctly reports nothing on
   identical inputs where the DFC fabricates.

---

## 7. The big caveat, and what's next

**Everything above is a synthetic concept model, not a real neural network.**
Whether any of it transfers to real models is the single largest open risk, and
it needs a GPU and real weights — neither of which this environment has
(`huggingface.co` is blocked here). `docs/HANDOFF.md` has the exact steps for
your machine.

Two concrete next steps:

- **Re-run the cross-architecture comparison at 10 seeds** (~25 min). It's
  currently 3-seed, and 3 seeds have already produced two conclusions that 10
  overturned. This is the finding the pivot rests on.
- **Validate on real weights** on your 8GB GPU — the null test first, since it's
  cheap and decides whether the toy geometry transfers.

---

## Where things live

| File | What's in it |
|---|---|
| `docs/RESULTS.md` | Corrected data and the current recommendation |
| `docs/CORRECTIONS.md` | Everything I got wrong, and why |
| `docs/DESIGN.md` | Architecture and methodology |
| `docs/HANDOFF.md` | Running it on your machine |
| `docs/RESEARCH-NOTES.md` | The paper, read in full |
| `runs/*.json` | Raw results; `runs/*.sh` reproduces them |
| `runs/contaminated/` | Superseded results, quarantined |

Branch: `claude/clever-hopper-icchb3` · 69 tests passing
