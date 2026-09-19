# Design

## 1. Problem

Given two models `A` and `B`, produce a ranked, causally-validated list of
*systematic differences* between them, without being told in advance what to
look for.

For our chosen use case, `B` is a transformed version of `A`: quantized
(int8 / int4 / NF4), merged with another checkpoint, or distilled. The question
a user brings is: **"I compressed my model. What did I lose?"**

The existing answer is perplexity delta plus a benchmark sweep. That measures
*average* degradation. It is close to blind to selective degradation, which is
the failure mode people actually fear — the compressed model is 99% as good and
1% worse in exactly the place that matters.

## 2. Non-goals

- Source-code diffs. Different problem, same word.
- A general-purpose interpretability library. There are good ones; we depend on
  them rather than rebuild them.
- Claims about frontier models. Everything here runs at ≤0.6B parameters. We
  will state which findings are scale-dependent rather than imply they transfer.
- Deployment/serving. This is an offline analysis tool that emits a report.

## 3. Ground truth — the core design bet

This is the part that distinguishes the project from a reimplementation.

The model-diffing literature validates qualitatively: show max-activating
examples for a discovered feature, write a plausible label, move on. That cannot
distinguish a real finding from an artifact of the sparsity penalty — and we
know from the crosscoder-artifact work that a large fraction of "model-exclusive"
latents in L1 crosscoders are artifacts.

Quantization and merging give us four independent checks that the CCP-alignment
style of result cannot support:

### 3.1 The null test
Diff a model against **itself**. A correct tool returns approximately nothing.

This is cheap, obvious, and — as far as we can tell from the published work —
not routinely reported. Any method with a nonzero exclusive-feature count on the
identity transform has a false-discovery rate we can measure directly, and every
downstream finding must be discounted by it. We run this first, before anything
else, and we report it in every experiment.

### 3.2 Monotonicity
The magnitude of the discovered diff should be ordered by how destructive the
transformation was:

```
diff(bf16, bf16)  ≈  0        # null
diff(bf16, int8)  <  diff(bf16, int4)
```

We know the ordering a priori. A method that inverts it, or that reports
comparable diffs for int8 and int4, is measuring noise.

### 3.3 Implanted-feature recovery
Fine-tune a small model to acquire a **known, narrow, behaviorally-checkable
capability** — e.g. a synthetic entity it must always mention in a specific
context. Verify behaviorally that the fine-tune has it. Then quantize
aggressively and check behaviorally whether the capability survived.

This gives a labelled retrieval task:
- If the capability was destroyed, the diff must surface it in the top-k.
- If the capability survived, the diff must *not* claim it as a difference.

This converts "did we find something interesting?" into a measurable
recall-at-k, which is what an industry buyer needs and what the literature
currently cannot provide.

### 3.4 Causal validation
For every reported feature, both models are runnable, so we can always test the
claim rather than assert it: ablate or steer along the feature direction and
measure the behavior delta on held-out prompts.

**Critically, against a random-feature control.** Ablating a random direction in
a sparse dictionary also changes behavior somewhat. A finding only counts if its
causal effect exceeds the random-direction baseline by a stated margin. Without
this control, causal validation is theater.

## 4. Architecture

```
diffdiff/
  models/        model loading; the transforms under test (quantize, merge)
  activations/   capture, disk-backed cache (memmap), streaming batch loader
  align/         token + layer correspondence between A and B  [see §5]
  diffing/
    llm_baseline.py   black-box control — no weights, no GPU
    crosscoder.py     standard L1 crosscoder
    batchtopk.py      BatchTopK crosscoder (the known artifact fix)
    dfc.py            Dedicated Feature Crosscoder — partitioned latent space
    trivial.py        logit KL, activation cosine — the "is this even needed" controls
  validate/
    null.py           §3.1
    monotonic.py      §3.2
    implant.py        §3.3
    causal.py         §3.4, with random-direction control
  report/          the deliverable artifact
```

Every method in `diffing/` implements one interface:

```python
def diff(A, B, corpus, budget) -> list[Finding]
```

where a `Finding` carries a direction (or, for the black-box baseline, a natural
language hypothesis plus discriminating prompts), a score, and enough metadata
for `validate/` to test it causally. The point of the uniform interface is that
the LLM baseline and DFC are scored by *identical* downstream machinery. If the
comparison is not apples-to-apples the whole exercise is worthless.

## 5. Alignment — deferred on purpose

Aligning activations between two models is the hard problem in cross-architecture
diffing. Different tokenizers segment text differently, so there is no canonical
token-to-token correspondence, and layer depth has no natural correspondence
either.

**For quantization and merging, both models share a tokenizer, architecture, and
layer count.** Alignment is free: token-for-token, layer-for-layer.

This is a deliberate reason to start here. We build the cache, the four diffing
methods, and the entire validation apparatus against an alignment layer that is
the identity — then swap in a real one.

`align/` therefore exists from day one as an interface with a trivial
implementation, so that the cross-architecture case (vendor migration) is a new
implementation rather than a rewrite. The planned cross-tokenizer strategy is
anchor-point pooling: find positions where both tokenizations agree on a
character boundary, pool activations within each span. This costs token-level
precision, which is exactly what makes features interpretable. We expect that
cost to be significant and we will measure it rather than hide it.

## 6. Compute budget (8GB VRAM)

The arithmetic, because it determines whether this is feasible at all:

| Item | Size |
|---|---|
| Two models @ 0.6B, bf16 | ~2.4 GB VRAM |
| Residual stream, d_model ≈ 896 | — |
| Activation cache, 2M tokens × 896 × fp16 × 2 models | ~7 GB **disk** |
| Crosscoder: 2 models × {enc,dec} × 32k dict × 896 | ~115M params |
| …in fp32 + Adam moments | ~1.8 GB VRAM |

Total VRAM during crosscoder training is well under 8GB because model weights
are not resident then — activations are read from the memmap cache. Capture and
training are separate passes for exactly this reason.

**Candidate models:** Qwen3-0.6B, SmolLM2-135M/360M, pythia-160m/410m, gpt2-small.
**Quantization:** bitsandbytes int8 and NF4 first (simplest path), GPTQ/AWQ later.

The LLM baseline runs through an API, so it consumes no VRAM. This makes it a
*cheap* control, which is precisely why it is dangerous to our thesis and must be
run honestly.

## 7. Metrics

| Metric | What it catches |
|---|---|
| Null-test false discovery rate | Sparsity artifacts masquerading as findings |
| Monotonicity (int8 vs int4 ordering) | Whether the diff magnitude is signal or noise |
| Implant recall @ k | Whether a known destroyed capability is actually surfaced |
| Implant false-positive rate | Whether a known *surviving* capability is wrongly flagged |
| Causal effect vs random-direction control | Whether reported features do anything |
| Cost per run (GPU-hours, API dollars) | Whether the white-box method earns its price |

The last row is not a formality. The entire commercial question is whether
mechanistic access buys enough over a black-box method to justify needing the
weights.

## 8. Phase plan

- **Phase 0 — scaffold + null test.** Model loading, transforms, and the null
  test running end to end on the smallest model. Deliberately first: the null
  test is the cheapest way to catch a broken harness.
- **Phase 1 — activation cache + LLM baseline.** Disk-backed capture, and the
  black-box control running end to end. We establish the bar before building the
  thing that must clear it.
- **Phase 2 — L1 crosscoder + BatchTopK.** Reproduce the known artifact and the
  known fix. If we cannot reproduce Complete Shrinkage on a transform where we
  know the answer, our harness is wrong.
- **Phase 3 — DFC.** The paper's contribution, scored against Phases 1–2.
- **Phase 4 — implant experiment + causal validation.** The evidence that makes
  any of this sellable.
- **Phase 5 — report artifact.** The thing a model-risk reviewer actually reads.

## 9. Kill criteria

Stated now, while it is cheap to be honest:

1. **If the LLM black-box baseline matches DFC** on implant recall and causal
   fidelity at this scale, then for *this* use case white-box access is not
   justified, and we should say so and pivot to the baseline. This is a real
   possibility — it is the published finding of at least one recent paper.
2. **If the trivial controls suffice** — if logit KL or activation cosine
   similarity on a prompt set localizes quantization damage as well as a trained
   crosscoder does — then the dictionary learning is unnecessary overhead.
3. **If the null test cannot be driven near zero** for any method, the
   false-discovery floor swamps the findings and the output is not auditable.

A negative result on (1) or (2) is a genuinely useful outcome and should be
written up, not buried. The failure mode to avoid is spending months building a
mechanistic pipeline that a prompting loop already matched.

## 10. Open questions

- Which layer(s) to diff? Quantization error accumulates with depth, so the
  answer is probably "several," which multiplies cache cost. Needs measurement.
- Does quantization damage even *look* like a sparse feature difference, or is it
  diffuse noise across all directions? If it is diffuse, dictionary methods are
  the wrong tool and the null/monotonicity tests will tell us early.
- Is a 0.6B model's feature geometry representative enough that conclusions
  transfer upward? We cannot answer this on 8GB. We should not pretend otherwise.
