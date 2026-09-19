# Research notes

## Provenance warning

`anthropic.com` and `arxiv.org` are both blocked by this environment's egress
policy, so the primary sources below were **not read directly**. These notes are
assembled from search-surfaced abstracts, summary/aggregator pages, and prior
knowledge of the crosscoder literature.

Everything marked **[unverified]** needs checking against the actual PDF before
we rely on it. Nobody should treat this file as a substitute for reading the
papers.

## The source material

**[Cross-Architecture Model Diffing with Crosscoders: Unsupervised Discovery of
Differences Between LLMs](https://arxiv.org/abs/2602.11729)** — Jiralerspong et
al., Feb 2026. An Anthropic Fellows project; the accompanying write-up is
[A "diff" tool for AI models](https://www.anthropic.com/research/diff-tool).

The chain of ideas:

1. **Model diffing** compares two models' internal representations to surface
   differences without being told what to look for. The motivation is that
   evaluations only test what their authors anticipated.
2. **Crosscoders** are the existing tool — a sparse dictionary trained jointly on
   both models' activations, yielding latents tagged A-only, B-only, or shared.
   See [Insights on Crosscoder Model Diffing](https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html).
3. Prior work targeted the **easy case**: a base model versus its own fine-tune.
   Same architecture, same tokenizer, free layer correspondence.
4. This paper targets **cross-architecture** diffing — different architectures,
   tokenizers, and training histories. Models analyzed: Llama-3.1-8B-Instruct,
   Qwen3-8B, GPT-OSS-20B, DeepSeek-R1-0528-Qwen3-8B.
5. The contribution is the **Dedicated Feature Crosscoder (DFC)**, which
   partitions the latent space by construction into A-exclusive / shared /
   B-exclusive rather than relying on an L1 penalty to separate them.

Reported findings: CCP-alignment features in Qwen3-8B and DeepSeek-R1-0528-Qwen3-8B,
American-exceptionalism features in Llama-3.1-8B-Instruct, and copyright-related
features. **[unverified]** — we have the claims, not the evidence.

## Why DFC exists

[Overcoming Sparsity Artifacts in Crosscoders to Interpret Chat-Tuning](https://arxiv.org/abs/2504.02922)
(Minder et al.) documents two failure modes of L1 crosscoders:

- **Complete Shrinkage** — if a concept is slightly more important in one model,
  the L1 penalty can drive the other model's representation to exactly zero,
  making a shared concept look exclusive.
- **Latent Decoupling** — the crosscoder learns separate A-only and B-only copies
  of the *same* concept, because under L1 this is mathematically equivalent to
  learning one shared latent.

Both manufacture fake "model-exclusive" features, which is the one thing model
diffing is supposed to produce. Their fix is a **BatchTopK** crosscoder enforcing
true L0 sparsity; they report that most L1 chat-only latents are not genuinely
chat-specific while most BatchTopK ones are.

DFC attacks the same problem architecturally rather than through the sparsity
mechanism. Imposing the partition instead of regularizing toward it is the more
robust instinct, and it removes a degree of freedom rather than adding a
hyperparameter — but it needs the same scrutiny, which is what our null test in
[`DESIGN.md`](DESIGN.md) §3.1 is for.

## The competing baseline — treat as a threat, not a footnote

[Simple LLM Baselines are Competitive for Model Diffing](https://arxiv.org/abs/2602.10371)
— Kempf, Schrodi, Cywiński, Brox, Nanda, Conmy, Feb 2026.
Code: https://github.com/eliaskempf/model-diffing

They propose evaluation metrics for generalization, interestingness, and
abstraction level, and report that an improved LLM-based baseline performs
comparably to SAE-based diffing while typically surfacing *more abstract*
behavioral differences.

Why this matters more for an industry use case than for a research one:

- The LLM baseline needs **no weights, no activations, and no GPUs**. It works
  against API-only models.
- Crosscoders need white-box access to **both** models plus a training run.
- Therefore the only buyers for a crosscoder-based product are those who hold
  both sets of weights: model hosts, fine-tuners, enterprises running
  open-weight models, and auditors with weight escrow.

Any pitch built on the DFC paper has to answer "why not just prompt both models
a lot?" — which is why the baseline is a first-class component of this repo
rather than a related-work citation.

## Open questions for when the PDFs are reachable

1. **How does the paper align activations across different tokenizers?** No
   summary explains this, and it is the crux of cross-architecture diffing. If
   the answer is sequence- or prompt-level pooling, that sacrifices the
   token-level precision that makes features interpretable — and it is our
   opening rather than a reason to walk away.
2. **How are the reported features validated?** If the evidence is
   max-activating examples plus auto-interp labels, it is a research artifact,
   not an audit artifact. If there is causal (ablation/steering) validation with
   controls, that changes our assessment substantially.
3. **Is a null test reported?** Does a DFC trained on a model against itself
   return nothing?
4. **How were layers chosen** for the cross-architecture pairs, and how sensitive
   are the results to that choice?
5. **What is the compute cost** of training a DFC at 8–20B scale? Determines
   whether the enterprise version is a per-run spend or a capital expense.

## Reading list

- https://arxiv.org/abs/2602.11729 — the source paper
- https://www.anthropic.com/research/diff-tool — the accessible write-up
- https://arxiv.org/abs/2602.10371 — the black-box baseline we must beat
- https://arxiv.org/abs/2504.02922 — crosscoder sparsity artifacts and BatchTopK
- https://transformer-circuits.pub/2025/crosscoder-diffing-update/index.html — crosscoder diffing background
- https://github.com/eliaskempf/model-diffing — baseline implementation
