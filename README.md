# diff-the-diff

An industry-oriented harness for **model diffing**: comparing the internal
representations of two language models to surface systematic differences that
benchmark evaluations never thought to test for.

This repository is not about source-code diffs. The "diff" here is the one from
mechanistic interpretability — see [`docs/RESEARCH-NOTES.md`](docs/RESEARCH-NOTES.md).

## The use case we are building for

**Quantization and merge verification.** You quantize a model to int4, or merge
two checkpoints, and you check perplexity and a few benchmarks. That catches
broad capability loss. It does not catch *selective* loss — a safety refusal
behavior degrading while perplexity barely moves, or a merge silently dropping
a capability that neither parent's eval suite covers.

We are building the tool that answers: **what actually changed inside the model,
and can you prove it?**

## Why this use case first

Quantization and merging are model transformations where we *approximately know
the right answer* — unlike comparing two unrelated models, where there is no way
to check whether a reported difference is real or how many were missed.

That makes them a setting where the method can be falsified rather than merely
demonstrated, and it makes four checks cheap that the source paper does not
report: a null test, a monotonicity test, a comparison against a black-box
baseline, and a characterization of the near-identical regime. See
[`docs/DESIGN.md`](docs/DESIGN.md) §3 and §6.

**Known risk:** the paper reports that this method struggles on base-vs-finetune
pairs due to "mirror features," and quantization is a more extreme version of
that regime. This is the project's central open question and Phase 0 is designed
to answer it cheaply — see [`docs/DESIGN.md`](docs/DESIGN.md) §11.

## Status

**Phase 0 complete**: the harness runs, and the null test has a result.

```bash
pip install -e ".[dev]"
pytest -q                                          # 50 tests
python -m diffdiff.cli null    --seeds 0 1 2       # diff a model against itself
python -m diffdiff.cli control --seeds 0 1 2       # positive control
```

| Document | Contents |
|---|---|
| [`docs/PHASE0-RESULTS.md`](docs/PHASE0-RESULTS.md) | What the null test found, and what it decides |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Architecture, ground-truth strategy, metrics, phase plan, kill criteria |
| [`docs/RESEARCH-NOTES.md`](docs/RESEARCH-NOTES.md) | The source paper, read in full, and what it does and does not establish |
| [`docs/HANDOFF.md`](docs/HANDOFF.md) | Running the real-model path on a machine with a GPU and model-hub access |

### What Phase 0 contains

- `diffdiff/diffing/` — BatchTopK crosscoder and DFC (dedicated partitions with
  structurally zeroed opposing decoders), with sparsity annealing and the AuxK
  dead-feature loss.
- `diffdiff/data/toy.py` — the paper's synthetic concept model (§2.3), extended
  with a perfect-null regime and a quantization-like near-identical regime.
- `diffdiff/validate/` — the null test and a mirror-pair detector, both scored
  against controls.
- `diffdiff/align/` — the paper's Algorithm 1 window-expansion aligner, for the
  cross-architecture case later.
- `diffdiff/models/`, `diffdiff/data/cache.py` — real-model loading,
  quantization, hooked activation capture, and a memmap cache.

## Constraints that shape everything here

- **8GB VRAM.** Models are capped at ~0.6B parameters. Activations are cached to
  disk, not held in VRAM. This is a real constraint, not a placeholder — see
  § Compute budget in the design doc for the arithmetic.
- **The black-box baseline is a first-class control, not an afterthought.** If a
  method that needs no weights and no GPU matches our white-box method, we need
  to know that before building a product on top of crosscoders.
