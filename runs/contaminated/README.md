# Contaminated results — do not cite

Every JSON in this directory was produced before the RNG collision fix
(`_DATA_SEED_OFFSET` in `diffdiff/data/toy.py`).

`ToyActivations` seeded its own `torch.Generator` with the run seed, while
callers seeded model initialisation with `torch.manual_seed(<same seed>)`. Same
generator algorithm, same seed, same stream — so `nn.init.normal_` on a decoder
drew the same numbers as the concept matrix, and the first rows of `W_dec_a`
were **exactly the ground-truth concepts**.

Consequences:

- **Concept-matching metrics were invalid on the A side.** An untrained model
  scored `best_cos = 1.0000` against ground truth.
- **Null-test floors were likely inflated.** The exclusive partition was
  initialised to real, frequent concepts, which are immediately useful for
  reconstruction and so stay alive and carry activation mass.
- **`standard_extreme_frac` is suspect**, because the contamination hit
  `W_dec_a` but not `W_dec_b`, and relative decoder norm is a ratio of the two.

These files are kept so the corrected numbers can be compared against them, not
because any conclusion drawn from them stands.
