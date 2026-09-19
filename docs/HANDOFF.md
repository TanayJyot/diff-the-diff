# Handoff: running this on your machine

Phase 0 was built and run in a sandbox with **no GPU, no Hugging Face access,
and no `download.pytorch.org`**. Everything in `diffdiff/` that does not touch
model weights is therefore tested and working; everything that does is written
but unexercised. This document says exactly which is which, and what to run.

## What you need

- An 8GB (or larger) CUDA GPU.
- Network access to `huggingface.co`.
- `pip install -e ".[models,dev]"`

## What is already verified

Run this anywhere, no GPU and no weights required:

```bash
pip install -e ".[dev]"
pytest -q                      # 47 tests
python -m diffdiff.cli null --seeds 0 1 2 --json runs/null.json
```

The null test runs entirely on the synthetic concept model
(`diffdiff/data/toy.py`, the paper's §2.3 design). Results in
[`docs/PHASE0-RESULTS.md`](PHASE0-RESULTS.md).

Verified by tests: the crosscoder and DFC (including the structural-zero
guarantee and the severed gradient path), BatchTopK, sparsity annealing, the
AuxK dead-feature loss, activation normalisation, outlier masking, the toy
generator's three regimes, mirror detection, the memmap cache, and the
window-expansion aligner — including the paper's own `['1989']` vs
`['198','9']` example, exercised against fake tokenizers.

## What is written but never executed

These need real weights and are the first thing to smoke-test on your machine:

| Module | What it does | Risk |
|---|---|---|
| `models/loader.py` | Loads a model, resolves the middle layer | `_transformer_blocks` guesses the attribute path per architecture; a model it doesn't know raises with a clear message |
| `models/transforms.py` | int8 / NF4 quantization via bitsandbytes | Untested against a real bitsandbytes install |
| `models/capture.py` | Hooks the residual stream, writes the cache | The forward hook assumes blocks return a tensor or a tuple whose first element is the hidden state |

## Step 1 — smoke-test the real-model path (~10 minutes)

The cheapest check that loading, hooking and caching work end to end:

```python
from diffdiff.models.loader import load_model
from diffdiff.models.transforms import apply_transform
from diffdiff.models.capture import capture_pair

base = load_model("HuggingFaceTB/SmolLM2-135M", layer="middle", device="cuda")
quant = apply_transform("nf4", "HuggingFaceTB/SmolLM2-135M", device="cuda")
print(base.layer, base.d_model)

texts = ["The quick brown fox jumps over the lazy dog. " * 20] * 50
cache_a, cache_b = capture_pair(
    base, quant, texts,
    "caches/base.bin", "caches/nf4.bin",
    n_tokens=50_000, transform="nf4",
)
```

If `_transformer_blocks` or the hook raises, that is the expected failure and
the fix is local to those two functions.

## Step 2 — the real null test (the decision point)

This is the experiment that decides whether the project continues on
quantization or pivots to cross-architecture. **Diff a model against itself**
— same weights, same tokenizer, identity transform:

```python
from diffdiff.data.cache import ActivationCache, paired_batch_fn
from diffdiff.diffing.crosscoder import Crosscoder, CrosscoderConfig
from diffdiff.diffing.train import TrainConfig, train, collect_features
from diffdiff.validate.mirror import find_mirror_pairs

# capture_pair(base, base, ...) — identity transform
a = ActivationCache.open("caches/base.bin")
b = ActivationCache.open("caches/base_copy.bin")
batch_fn = paired_batch_fn(a, b, device="cuda")

cfg = CrosscoderConfig(d_a=a.meta.d_model, d_b=b.meta.d_model,
                       n_features=16384, exclusive_frac=0.05, k=32, k_initial=128)
dfc = Crosscoder(cfg)
train(dfc, batch_fn, TrainConfig(steps=20_000, batch_size=512, device="cuda"))
feats = collect_features(dfc, batch_fn, n_batches=40, batch_size=512, device="cuda")
print(find_mirror_pairs(dfc, feats).summary())
```

Run it for at least three seeds. The paper found its broad feature in every run
but granular ones in only 2–3 of 5, so a single seed tells you nothing.

**Read the result against the toy-model baseline** in
[`docs/PHASE0-RESULTS.md`](PHASE0-RESULTS.md). If real activations behave like
the synthetic ones, the conclusion there transfers. If they diverge, that
divergence is itself the finding, and it means the toy model is missing
something about real feature geometry.

## Step 3 — monotonicity

Only worth running if step 2 does not disqualify the regime:

```
diff(bf16, bf16)  ~  0        # step 2
diff(bf16, int8)  <  diff(bf16, nf4)
```

`DAMAGE_ORDER` in `models/transforms.py` encodes the expected ordering. A method
that inverts it, or that reports comparable diffs for int8 and NF4, is measuring
noise rather than damage.

## Budget notes

Scaled from the paper's figures (3× H100 for ~24h of activation collection,
1× H100 for ~24h per crosscoder, 131,072 features, 100M tokens):

- **Disk, not VRAM, is the binding constraint on capture.** 2M tokens × 576 dims
  × fp16 × 2 models ≈ 4.4GB. SmolLM2-135M is the cheapest useful starting point.
- **Weights are not resident during crosscoder training** — activations stream
  from the memmap — so a 16k-feature dictionary at `d_model=576` trains in
  roughly 1GB of VRAM plus the batch.
- **Batch size trades against memory directly**, because BatchTopK holds
  pre-activations for the whole batch: 512 × 16,384 × fp32 is 34MB before the
  backward pass. Drop it before dropping dictionary size.
- **Auto-interp is the real cost driver**, not GPU time — the paper reports
  ~500,000 Claude queries per experiment. We only need it once features must be
  *explained* (Phase 4), not to run the null test. Budget it per seed.

## What has not been built yet

Phase 0 stops at the null test, deliberately, because its result determines
whether the rest is worth building. Still to come, in order:

1. The LLM black-box baseline (Phase 1) — the control we must beat.
2. DSF crosscoder (Phase 2) — the paper's other baseline.
3. Model stitching and the exclusivity score (Phase 4) — their
   crosscoder-independent validation pathway.
4. The implant experiment (Phase 4) — our answer key.

See [`DESIGN.md`](DESIGN.md) §9.
