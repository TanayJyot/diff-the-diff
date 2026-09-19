# Research notes

Status: **paper read in full** (75pp, v1, 12 Feb 2026). The open questions from
the first draft of this file are answered below. The blog post is still
unreachable from this environment, but it is a summary of the paper and nothing
here depends on it.

## The source material

**[Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of
Differences Between LLMs](https://arxiv.org/abs/2602.11729)** — Thomas
Jiralerspong (Anthropic Fellow, Mila / Université de Montréal) and Trenton
Bricken (Anthropic).

Model diffing compares two models' internal representations to surface
differences without being told what to look for — aimed at **unknown unknowns**,
behaviors no eval suite was written to catch. The paper's own framing: model
diffing "could have hypothetically flagged the 'overly sycophantic' behavior in
April's GPT-4o update, enabling a fix before it reached the general public."
Model diffing was used in the Claude 4.5 Sonnet system card to track
evaluation-aware behavior across training.

Prior crosscoder work only ever ran on base-vs-finetune pairs. This paper is the
first application to **cross-architecture** pairs.

## What a DFC actually is

A standard crosscoder learns a shared dictionary across both models and then
decides exclusivity **post hoc**, via Relative Decoder Norm
`R_i = ||d_i^A|| / (||d_i^A|| + ||d_i^B||)`, where `R_i ≈ 1` means A-exclusive.
The flaw: the joint reconstruction objective actively rewards features that
serve both models, so there is "an inherent optimization prior favoring shared
features." Exclusive features are also *learned later in training* than shared
ones (their Figure 15) — direct empirical evidence of that prior.

The **Dedicated Feature Crosscoder** partitions feature indices into three
disjoint sets `I_A`, `I_B`, `I_S` and **structurally constrains the opposing
decoder weights to zero**, so a feature in `I_A` satisfies `||d_i^B||₂ = 0`
*exactly* rather than approximately. Model A is reconstructed from `I_A ∪ I_S`,
Model B from `I_B ∪ I_S`. Severing the gradient flow removes the pressure on
exclusive features to become shared.

The partition fraction is a **hyperparameter**, and a consequential one — see
seed/partition sensitivity below.

## Answers to our five open questions

### 1. How are activations aligned across different tokenizers?

**A greedy window-expansion algorithm over decoded text** (Algorithm 1, §B.1.6),
not pooling. Walk both token sequences with two pointers; skip non-content
tokens; try a 1-to-1 decoded-string match; on mismatch, asymmetrically grow the
window on whichever side has the shorter decoded text until the decoded strings
agree.

Their example: Llama has `['1989']`, Qwen has `['198', '9']` → expand Qwen's
window → `decode('1989') == decode('198','9')` → matched.

**Critically, on a match they keep only the activation of the *final token* of
each window.** This "many-to-one compression" preserves a strict 1:1 activation
mapping, on the stated hope "that the final token's activation captures the
semantic context of the entire window."

This is better than the prompt-level pooling I assumed, and it answers my worry
about losing token-level precision — the precision is preserved by construction.
Reported alignment success is **>99%** on 1,000 FineWeb/LMSYS samples (99.2% for
Llama/Qwen, 8 failures). Failures concentrate in chat/conversational text (87.5%)
and special characters — box-drawing glyphs, smart quotes, multi-codepoint
emoji, code snippets.

*Our read:* the last-token assumption is load-bearing and untested in the paper.
It is a cheap, high-value thing for us to probe, and it is the most likely place
for cross-tokenizer diffing to quietly degrade.

### 2. How are features validated?

Much more rigorously than I guessed. Three independent layers:

**(a) Steering-vector transfer, to validate the shared space itself** (§3.1).
They take persona vectors ("evil", "hallucinating", "sycophantic") derived by a
method entirely independent of any SAE/crosscoder, confirm these are novel
directions (max cosine similarity against all 131,072 feature decoders was only
0.38 / 0.35 / 0.26), then translate Llama→Qwen through the shared partition and
steer. The transferred sycophantic vector reproduces sycophantic behavior in
Qwen. That is a genuinely strong test of alignment quality.

**(b) The exclusivity score, with a crosscoder-independent pathway** (§3.2.2).
Train a linear **model-stitching** map `T_{A→B}` on held-out data — a best-effort
translation that never touches the crosscoder dictionary. Project the candidate
feature's decoder vector through it, steer Model A with the original and Model B
with the stitched projection, and have an LLM judge (Claude 4.1 Opus) rate
behavioral similarity 1–5. `exclusivity = 6 − similarity`. A score of 5 means the
effect **could not be transferred even by a directly trained linear map**. 25
judge ratings were manually checked against human judgment.

**(c) Causal steering with specificity controls** (§3.3.1). Steer across 30
curated prompts at varying strengths, measuring both ideological alignment and
coherence. The key control: the Qwen CCP feature moves Qwen and has "almost no
effect" on Llama, and vice versa for Llama's American-exceptionalism feature.
They further steer with prompts about *other* countries to confirm the features
are country-specific rather than generic propaganda directions.

So my earlier "probably just max-activating examples" was **wrong**, and I should
correct that: auto-interp on max-activating examples is only the *screening*
stage, explicitly followed by causal validation.

Two caveats worth carrying forward. The exclusivity-score pool is **pre-filtered**
to features that are already alive, interpretable, and show clear steering — so
the reported distribution is conditional on that filter, not on all exclusive
features. And a Claude judge appears at three separate points in the pipeline.

### 3. Is a null test reported?

**No.** There is no diff of a model against itself. Our §3.1 null test remains an
unreported check, and given finding #6 below it is the single highest-value
experiment we can run cheaply.

### 4. How were layers chosen, and how sensitive are results?

**Middle-layer residual stream, chosen by convention, with no sensitivity
analysis.** Concretely: layer 16 of DeepSeek-R1-0528-Qwen3-8B against layer 12 of
GPT-OSS-20B (activation dims 4096 and 2880). No ablation over layer choice is
reported. This remains genuinely open, and our §10 question about diffing several
layers stands.

### 5. What does it cost?

| Item | Cost |
|---|---|
| Activation collection, per model pair | 3× H100 80GB, ~24h |
| Crosscoder training, per run | 1× H100 80GB, ~24h |
| Runs for the final paper | 5 crosscoders |
| Auto-interp | **~500,000 Claude 4.1 Opus queries per experiment** |

Training data: 100M token-aligned activation pairs, 50/50 FineWeb +
LMSYS-Chat-1M. Dictionary 131,072 (expansion factor 32), BatchTopK k=200,
Adam @ 1e-4, batch 2048, 100k steps, bf16.

The striking line is the API cost, not the GPU cost. **Auto-interp is the
budget driver**, and it scales with dictionary size — which is the constraint we
should design around, not VRAM.

## Six findings that change our plan

1. **They already use BatchTopK.** k=200, following Minder et al. So the L1
   artifacts (Complete Shrinkage, Latent Decoupling) are already mitigated in
   their baseline; DFC is a fix for a *different* problem — the shared-feature
   optimization prior. Our Phase 2 should treat BatchTopK as the baseline, not as
   a contribution.

2. **DFC buys recall and pays in precision.** On their synthetic toy model with
   ground-truth concepts, DFC beats both the standard crosscoder and the
   Designated Shared Feature (DSF) crosscoder on exclusive-concept recall — "at
   the cost of an increased false-positive rate." They argue this trade is right
   for safety auditing, since a false negative costs more than a false positive.
   **For an enterprise audit product that argument partly inverts**: false
   positives are alert fatigue, and alert fatigue is how tools get switched off.

3. **The advantage is largest in the undercomplete regime** (dictionary smaller
   than the number of true concepts) — which they note "is most likely the regime
   in which real-world applications operate," and which is unavoidably *our*
   regime on 8GB. Good news for small-scale work.

4. **Most exclusive features are not meaningful.** Stated plainly: "most features
   identified as model-exclusive do not capture meaningful behavioral
   differences." They therefore propose crosscoder diffing "primarily as a
   **high-recall pre-screening tool**" inside a **"screen-and-verify" workflow**.
   This is the honest framing and we should adopt it rather than oversell.

5. **Discovery is unstable across seeds and partition sizes.** The broad CCP
   feature was found in every run; American exceptionalism in 4/5; the granular
   pro-China narratives in only 2–3/5. Smaller partitions (1%, 3%) found the
   broad features but missed granular ones. **Any metric we report must be over
   multiple seeds** — single-run results are not meaningful for this method.

6. **The method struggles in exactly our regime.** §5.2, verbatim: "our method
   can struggle in certain contexts, such as **base vs. finetune comparisons**,
   where, in preliminary experiments, we encountered the **'mirror features'**
   phenomenon noted in prior work." They call a full investigation of why "a
   promising direction for future work."

   Quantization and merging are the *extreme* near-identical case — far closer
   than base-vs-finetune. A DFC with a 5% partition must fill 6,554 exclusive
   slots with something; when the two models are nearly identical there is little
   genuinely exclusive to find, so the partition plausibly fills with mirror
   pairs. **This is a direct hit on the use case we picked.** See `DESIGN.md` §11.

## Related methods worth knowing

- **DSF (Designated Shared Feature) crosscoder**, Mishra-Sharma et al. — the
  near-dual of DFC: dedicates features to be explicitly *shared* via decoder
  weight-sharing. The paper adapts it from L1 to BatchTopK and uses it as a
  baseline. We should too.
- **Model stitching**, Chen et al. 2025a — the independent linear `A→B` map that
  makes the exclusivity score credible. Cheap, and useful to us on its own.
- **Persona vectors**, Chen et al. 2025b — independently-derived behavioral
  directions, the input to the transfer test.
- [Overcoming Sparsity Artifacts in Crosscoders](https://arxiv.org/abs/2504.02922)
  — Complete Shrinkage and Latent Decoupling, and BatchTopK as the fix.
- [Simple LLM Baselines are Competitive for Model Diffing](https://arxiv.org/abs/2602.10371)
  — the black-box control. Still first-class for us; note this paper does **not**
  compare against it.

## Engineering details worth copying

Small, hard-won things we should not rediscover ourselves:

- **Activation normalization**: scale median L2 norm to `sqrt((d1+d2)/2)` so both
  models contribute comparably.
- **Outlier masking**: drop any activation whose L2 norm exceeds 2× the batch
  median from the loss. They note Qwen produced norms up to **10× the median**,
  especially at the first token position. This will bite us too.
- **Sparsity annealing**: anneal k from 1000 → 200 over the first 5,000 steps so
  features form before sparsity is enforced.
- **AuxK dead-feature loss** (Gao et al.), α=0.03; a feature counts as dead after
  10M consecutive tokens without activating.
- Initial decoder norm scale 0.4; gradient checkpointing; bf16.

## Their own stated limitations

- Validating exclusivity on real models without ground truth is "inherently
  challenging"; the exclusivity score "is a proxy, but is not definitive."
- Discovery is probabilistic and sensitive to partition size and initialization.
- Generalizability across more model pairs is untested.
- Mirror features in base-vs-finetune (finding #6).
- **No claims about provenance** — whether a feature came from explicit training
  or a data artifact is not addressed. For an audit product, provenance is
  usually the first question a customer asks.
- §5.1: "model-exclusive" is a claim about the **representation, not the
  concept**. Qwen having an exclusive CCP-alignment direction does not mean Llama
  lacks the concept — only that it has no direct *linear analogue* in Llama's
  activation space. Any report we generate must make this distinction, or it will
  be read as a much stronger claim than the method supports.
