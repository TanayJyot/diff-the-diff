# Design

> Revised after reading the paper in full. The substantive change is §11: the
> paper reports that this method struggles in precisely the regime our use case
> occupies. That is now the project's central open risk.

## 1. Problem

Given two models `A` and `B`, produce a ranked, causally-validated list of
*systematic differences* between them, without being told in advance what to
look for.

For our use case `B` is a transformed version of `A`: quantized (int8 / int4 /
NF4), merged with another checkpoint, or distilled. The user's question is
**"I compressed my model. What did I lose?"**

The standard answer is perplexity delta plus a benchmark sweep, which measures
*average* degradation and is close to blind to *selective* degradation — the
compressed model is 99% as good, and 1% worse in exactly the place that matters.

## 2. Non-goals

- Source-code diffs. Different problem, same word.
- A general-purpose interpretability library. We depend on existing ones.
- Claims about frontier models. Everything here runs at ≤0.6B parameters, versus
  the paper's 8–20B. We state which findings are scale-dependent rather than
  imply they transfer.
- Deployment/serving. This is an offline analysis tool that emits a report.

## 3. What the paper already validates, and what it doesn't

An earlier draft of this document asserted that the model-diffing literature
validates only qualitatively. Having read the paper, that was wrong and is
corrected here — it matters, because it changes what our contribution is.

**Already done well, and we should adopt rather than reinvent:**

- A **synthetic toy model with ground-truth concepts**, measuring exclusive-concept
  recall *and* false-positive rate against standard and DSF crosscoders.
- An **exclusivity score** validated through a pathway independent of the
  crosscoder: train a linear model-stitching map `T_{A→B}`, push the feature
  through it, steer both models, judge behavioral similarity. A max score means
  the effect could not be transferred even by a directly trained linear map.
- **Causal steering with specificity controls** — the feature moves its own model
  and not the other, and is specific to its topic rather than a generic direction.
- Honest framing: crosscoder diffing is a **high-recall pre-screening tool**
  inside a **screen-and-verify** workflow, not an oracle.

**Genuinely missing, which is where we can add something:**

1. **No null test.** The paper never diffs a model against itself. With a forced
   partition that *must* be filled, this is the obvious falsification test and it
   is not reported.
2. **No monotonicity test.** No setting where the size of the true difference is
   known in advance and the method's output can be checked for correct ordering.
3. **No comparison against a black-box baseline.** The competing claim that a
   simple LLM method is competitive goes unaddressed.
4. **No characterization of the near-identical regime** beyond noting that it
   fails (§11).
5. **No layer-choice sensitivity analysis.** Middle layer by convention.

Our contribution is (1)–(4). Quantization is the setting that makes all four
cheap, because the size and nature of the true difference are known a priori.

## 4. Architecture

```
diffdiff/
  models/        model loading; the transforms under test (quantize, merge)
  activations/   capture, disk-backed cache (memmap), streaming batch loader
  align/         token + layer correspondence between A and B  [see §5]
  diffing/
    llm_baseline.py   black-box control — no weights, no GPU
    crosscoder.py     BatchTopK crosscoder (the paper's own baseline)
    dsf.py            Designated Shared Feature crosscoder — the near-dual of DFC
    dfc.py            Dedicated Feature Crosscoder — partitioned latent space
    trivial.py        logit KL, activation cosine — the "is this even needed" controls
  validate/
    null.py           §6.1 — diff a model against itself
    monotonic.py      §6.2 — int8 vs int4 ordering
    mirror.py         §11 — detect manufactured mirror-feature pairs
    stitch.py         model stitching + exclusivity score (from the paper)
    implant.py        §6.3 — recover a known implanted capability
    causal.py         §6.4 — steering, with a random-direction control
  report/          the deliverable artifact
```

Every method in `diffing/` implements one interface:

```python
def diff(A, B, corpus, budget) -> list[Finding]
```

A `Finding` carries a direction (or, for the black-box baseline, a natural
language hypothesis plus discriminating prompts), a score, and enough metadata
for `validate/` to test it causally. The point of the uniform interface is that
the LLM baseline and DFC are scored by *identical* downstream machinery — if the
comparison is not apples-to-apples the exercise is worthless.

## 5. Alignment — deferred on purpose, and now with a known-good design

For quantization and merging both models share a tokenizer, architecture, and
layer count, so alignment is the identity. `align/` still exists from day one as
an interface, so the cross-architecture case is a new implementation rather than
a rewrite.

When we do implement it, the paper's Algorithm 1 is the design to copy, and it is
better than the pooling approach this document originally assumed. Two pointers
walk both token sequences; non-content tokens are skipped; a decoded 1-to-1 match
is tried first; on mismatch the window grows asymmetrically on the side with the
shorter decoded text until the decoded strings agree. On a match, **only the
final token's activation in each window is kept**, preserving a strict 1:1
mapping. Reported success >99%, with failures concentrated in chat text, smart
quotes, box-drawing characters, and multi-codepoint emoji.

The load-bearing and untested assumption is that the final token's activation
carries the semantic content of the whole window. That is cheap for us to probe
and is the most likely place for cross-tokenizer diffing to quietly degrade.

## 6. Ground truth — four checks

### 6.1 The null test
Diff a model against **itself**. A correct tool returns approximately nothing.

The paper does not report this, and a DFC's exclusive partition is a fixed
fraction of the dictionary that the objective will fill regardless of whether
anything exclusive exists. So the null test measures the false-discovery floor
directly, and every downstream finding must be discounted by it. **We run this
first, before anything else, and report it in every experiment.**

**The null must be run in both capacity regimes, because they have different
incentives.** When the dictionary is *overcomplete* relative to the number of
concepts in the data, the shared partition alone has enough capacity to explain
everything, so the optimizer leaves the exclusive partitions largely dead. When
the dictionary is *undercomplete*, shared capacity is scarce, there is real
pressure to recruit exclusive features, and the false-discovery floor rises.

Measured (corrected, see `PHASE0-RESULTS.md`): undercomplete 0.0315 against
overcomplete 0.0012, a 25x gap.

> **History.** An earlier revision of this section claimed the opposite — that
> overcomplete dictionaries give roughly *twice* the floor — and proposed a
> mechanism in which scarcity suppresses the artifact. That was measured on data
> produced before the RNG collision fix (`_DATA_SEED_OFFSET` in
> `diffdiff/data/toy.py`), where decoders were initialised to the ground-truth
> concepts themselves and so stayed alive and carried mass. The contaminated
> overcomplete floor was 0.0740; the corrected one is 0.0012, a 60x difference.
> The prediction recorded here originally was correct and is restored. The note
> is kept because the retraction matters more than the tidy text would.

**Dictionary size remains a false-discovery knob**, but in the direction that
favours large dictionaries for diffing, not against them.

**A null result is meaningless without a positive control.** If the exclusive
partitions also stay empty in a regime where differences genuinely exist, then
the method is not finding anything at this scale and a clean null is vacuous
rather than reassuring. `diffdiff.cli control` runs the same measurement code on
a regime with planted exclusive concepts, for exactly this reason.

### 6.2 Monotonicity
The magnitude of the discovered diff should be ordered by how destructive the
transformation was:

```
diff(bf16, bf16)  ≈  0        # null
diff(bf16, int8)  <  diff(bf16, int4)
```

We know the ordering a priori. A method that inverts it, or reports comparable
diffs for int8 and int4, is measuring noise.

### 6.3 Implanted-feature recovery
Fine-tune a small model to acquire a **known, narrow, behaviorally-checkable
capability**. Verify behaviorally that it has it. Quantize aggressively, and
check behaviorally whether the capability survived. This gives a labelled
retrieval task: if the capability was destroyed the diff must surface it in the
top-k; if it survived the diff must not flag it.

### 6.4 Causal validation
Steer or ablate along the feature direction and measure behavior delta on
held-out prompts — **against a random-direction control**, since ablating a random
direction in a sparse dictionary also perturbs behavior. Without that control,
causal validation is theater.

We additionally adopt the paper's **exclusivity score** (model stitching + LLM
judge), because it tests exclusivity through a pathway that does not involve the
crosscoder at all. Note its pool is pre-filtered to alive/interpretable/steerable
features; we should report the filter's selectivity alongside the score, which
the paper does not.

## 7. Compute budget (8GB VRAM)

The paper's scale, for reference: 3× H100 for ~24h to collect activations,
1× H100 for ~24h per crosscoder, 5 crosscoders, 100M token-aligned activation
pairs, dictionary 131,072 (expansion factor 32), k=200, batch 2048, 100k steps.

Scaled to our hardware:

| Item | Paper | Ours |
|---|---|---|
| Model size | 8–20B | ≤0.6B |
| `d_model` | 4096 / 2880 | 576–1024 |
| Dictionary | 131,072 (×32) | 16k–32k (×16–32) |
| Activation pairs | 100M tokens | 2–5M tokens |
| Activation cache | — | ~8–20 GB **disk** |
| Crosscoder params | — | ~134M → ~2.1 GB VRAM w/ Adam |
| Batch | 2048 | 256–512 (BatchTopK holds all pre-activations) |

VRAM during crosscoder training stays under 8GB because model weights are not
resident — activations stream from the memmap cache. Capture and training are
separate passes for exactly this reason. The batch-size reduction is forced:
BatchTopK needs pre-activations for the entire batch, and 2048 × 32,768 × fp32 is
268MB before the backward pass.

**The real budget constraint is not VRAM.** The paper reports ~500,000 Claude 4.1
Opus queries per experiment for auto-interp. That scales with dictionary size, so
our 16k–32k dictionary and our decision to auto-interp only the exclusive
partition plus a sampled shared control should put us in the low thousands of
queries per run. This needs to be budgeted explicitly, per seed, because §7 also
requires multiple seeds.

Adopt from the paper without rediscovering: median-L2 activation normalization to
`sqrt((d1+d2)/2)`; outlier masking above 2× batch median norm (they saw 10×
median norms in Qwen at the first token position); sparsity annealing from
k=1000 → 200 over 5,000 steps; AuxK dead-feature loss at α=0.03 with a 10M-token
dead threshold; initial decoder norm scale 0.4; bf16 + gradient checkpointing.

**Candidate models:** Qwen3-0.6B, SmolLM2-135M/360M, pythia-160m/410m, gpt2-small.
**Quantization:** bitsandbytes int8 and NF4 first, GPTQ/AWQ later.

The LLM baseline runs through an API and consumes no VRAM, which makes it a cheap
control — and therefore a dangerous one for our thesis, which is why it must be
run honestly.

## 8. Metrics

| Metric | What it catches |
|---|---|
| Null-test false discovery rate | A partition filled with artifacts |
| Mirror-pair fraction (§11) | Exclusive features that are duplicates, not differences |
| Monotonicity (int8 vs int4 ordering) | Whether diff magnitude is signal or noise |
| Implant recall @ k | Whether a known destroyed capability is surfaced |
| Implant false-positive rate | Whether a known *surviving* capability is wrongly flagged |
| Causal effect vs random-direction control | Whether reported features do anything |
| Exclusivity score (stitching + judge) | Exclusivity via a crosscoder-independent path |
| Cost per run (GPU-hours, API dollars) | Whether white-box access earns its price |

**Every metric is reported over ≥3 seeds.** The paper found its broad feature in
every run but granular features in only 2–3 of 5, so single-run results are not
meaningful for this method and we should not produce any.

## 9. Phase plan

- **Phase 0 — scaffold + null test.** Model loading, transforms, and the null
  test end to end on the smallest model. Deliberately first: it is the cheapest
  way to catch a broken harness, and per §11 it is also our main scientific
  result if DFC turns out to be unusable in this regime.
- **Phase 1 — activation cache + LLM baseline.** Disk-backed capture, and the
  black-box control running end to end. Establish the bar before building the
  thing that must clear it.
- **Phase 2 — BatchTopK crosscoder + DSF.** The paper's own baselines. Note
  BatchTopK is *their* baseline too, so this is the real comparison point, not L1.
- **Phase 3 — DFC + mirror-feature analysis.** The contribution, scored against
  Phases 1–2, with §11 as the explicit object of study.
- **Phase 4 — implant experiment + causal validation.** The evidence that makes
  any of this sellable.
- **Phase 5 — report artifact.** What a model-risk reviewer actually reads.

## 10. Open questions

- Which layer(s) to diff? The paper uses the middle-layer residual stream by
  convention with no sensitivity analysis, so this is open in the literature too.
  Quantization error accumulates with depth, so the answer is probably "several,"
  which multiplies cache cost.
- Does quantization damage look like a sparse feature difference at all, or is it
  diffuse noise across all directions? If diffuse, dictionary methods are the
  wrong tool — and the null and monotonicity tests will tell us early and cheaply.
- Is a 0.6B model's feature geometry representative enough to transfer upward? We
  cannot answer this on 8GB and should not pretend otherwise.
- Does the final-token assumption in Algorithm 1 hold? Only matters once we do
  cross-architecture, but it is cheap to probe.

## 11. The central risk: mirror features in the near-identical regime

The paper's §5.2 states that the method "can struggle in certain contexts, such
as **base vs. finetune comparisons**, where, in preliminary experiments, we
encountered the **'mirror features'** phenomenon noted in prior work," and calls a
full investigation "a promising direction for future work."

**Quantization and merging are a more extreme version of that regime than
base-vs-finetune.** A quantized model differs from its parent by rounding error;
a base/finetune pair differs by a training run. So we have chosen the setting the
paper flags as its known failure mode, and we should be explicit about the
mechanism rather than hope:

A DFC allocates a fixed fraction of the dictionary to each exclusive partition —
5% of 131,072 is 6,554 features. That capacity exists whether or not anything
genuinely exclusive exists to fill it. When the two models are nearly identical,
the plausible outcome is that the partitions fill with **mirror pairs**:
duplicate representations of the same shared concept, one in each exclusive
partition, which look like differences and are not.

This has three consequences for the plan:

1. **The null test is promoted from sanity check to headline experiment.** It
   measures exactly this, it is the cheapest thing we can run, and Phase 0 will
   tell us whether the rest of the plan survives.
2. **We need a mirror detector** (`validate/mirror.py`): for each A-exclusive
   feature, look for a B-exclusive partner whose stitched projection is highly
   similar. Mirror pairs should be detectable and quantifiable, and the
   mirror-pair fraction becomes a first-class reported metric.
3. **The project's framing shifts, favorably.** This stops being "apply DFC to
   quantization" — a reimplementation — and becomes "characterize and mitigate
   the near-identical regime," which is an open problem the paper explicitly
   flags. If we can make DFC work where it currently fails, that is a real
   contribution rather than an application.

The alternative, if Phase 0 shows the regime is hopeless, is to pivot to the
cross-architecture use case (vendor migration) where the method demonstrably
works. **Phase 0 is cheap and decides this**, which is a good reason to run it
before building anything else.

## 12. Kill criteria

Stated now, while it is cheap to be honest:

1. **If the null test cannot be driven near zero**, the false-discovery floor
   swamps the findings and the output is not auditable. Given §11 this is the
   most likely failure, and it is a publishable negative result about the
   near-identical regime rather than a dead end.
2. **If the LLM black-box baseline matches DFC** on implant recall and causal
   fidelity at this scale, then for this use case white-box access is not
   justified and we should say so. The paper does not compare against it.
3. **If the trivial controls suffice** — if logit KL or activation cosine
   similarity localizes quantization damage as well as a trained crosscoder —
   the dictionary learning is unnecessary overhead.

A negative result on any of these is useful and should be written up, not buried.
The failure mode to avoid is spending months on a mechanistic pipeline that a
prompting loop already matched.
