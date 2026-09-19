"""Command line entry point.

Phase 0 exposes one command, ``null``, which runs the null test over several
seeds. See ``docs/DESIGN.md`` §6.1 and §11 for what it decides.
"""

from __future__ import annotations

import argparse
import json
import sys

from diffdiff.data.toy import cross_arch_config, null_config, quantized_config
from diffdiff.diffing.crosscoder import CrosscoderConfig
from diffdiff.diffing.train import TrainConfig
from diffdiff.validate.null import NullReport, run_control_test, run_null_test


def _add_null_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2],
                   help="seeds to run; the paper's own results vary run to run, "
                        "so single-seed numbers are not meaningful")
    p.add_argument("--d", type=int, default=128, help="activation dimension")
    p.add_argument("--concepts", type=int, default=256, help="ground-truth shared concepts")
    p.add_argument("--features", type=int, default=4096, help="dictionary size")
    p.add_argument("--exclusive-frac", type=float, default=0.05,
                   help="fraction of the dictionary dedicated to EACH exclusive partition")
    p.add_argument("--k", type=int, default=16, help="final BatchTopK sparsity")
    p.add_argument("--aux-alpha", type=float, default=0.03,
                   help="weight of the AuxK dead-feature loss. Set 0 to disable: the "
                        "loss revives features that stopped firing, so it may itself "
                        "be what populates an exclusive partition that should be empty")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--mirror-threshold", type=float, default=0.5)
    p.add_argument("--eval-batches", type=int, default=20)
    p.add_argument("--no-standard", action="store_true",
                   help="skip the standard-crosscoder reference run")
    p.add_argument("--device", default="cpu")
    p.add_argument("--json", metavar="PATH", help="write results as JSON")
    p.add_argument("--quiet", action="store_true")


def _toy_for(args: argparse.Namespace, seed: int):
    """Build the activation source for whichever regime was requested."""
    if args.command == "null":
        return null_config(d=args.d, n_shared=args.concepts, seed=seed)
    if args.regime == "cross-arch":
        return cross_arch_config(
            d_a=args.d, d_b=args.d, n_shared=args.concepts - args.n_excl,
            n_excl=args.n_excl, seed=seed,
        )
    offset = getattr(args, "drop_offset", 0)
    return quantized_config(
        d=args.d, n_shared=args.concepts, noise_std=args.noise,
        drop_concepts=tuple(range(offset, offset + args.n_excl)), seed=seed,
    )


def _run_regime(args: argparse.Namespace) -> int:
    label = "null test" if args.command == "null" else f"control ({args.regime})"
    runner = run_null_test if args.command == "null" else run_control_test
    reports: list[NullReport] = []
    for seed in args.seeds:
        if not args.quiet:
            print(f"\n=== {label}, seed {seed} ===", flush=True)
        report = runner(
            toy=_toy_for(args, seed),
            crosscoder=CrosscoderConfig(
                d_a=args.d, d_b=args.d, n_features=args.features,
                exclusive_frac=args.exclusive_frac, k=args.k, k_initial=args.k * 4,
                anneal_steps=min(5000, args.steps // 4),
                aux_alpha=args.aux_alpha,
            ),
            training=TrainConfig(
                steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                device=args.device, seed=seed,
            ),
            eval_batches=args.eval_batches,
            compare_standard=not args.no_standard,
            mirror_threshold=args.mirror_threshold,
            seed=seed,
            progress=None if args.quiet else lambda m: print(m, flush=True),
        )
        reports.append(report)
        if not args.quiet:
            print(report.summary(), flush=True)

    print(f"\n=== summary across seeds ({label}) ===")
    for r in reports:
        print(f"  seed {r.seed}: mass={r.exclusive_mass_frac:.4f} "
              f"mirror_excess={r.mirror.excess:+.3f} -> {r.verdict()}")

    n = len(reports)
    mean_mass = sum(r.exclusive_mass_frac for r in reports) / n
    mean_excess = sum(r.mirror.excess for r in reports) / n
    print(f"  mean exclusive mass {mean_mass:.4f} | mean mirror excess {mean_excess:+.3f}")

    if args.json:
        payload = [
            {
                "seed": r.seed,
                "verdict": r.verdict(),
                "exclusive_mass_frac": r.exclusive_mass_frac,
                "exclusive_live_frac": r.exclusive_live_frac,
                "expected_mass_frac": r.expected_mass_frac,
                "mirror_frac": r.mirror.mirror_frac,
                "mirror_control_frac": r.mirror.control_frac,
                "mirror_excess": r.mirror.excess,
                "mirror_median_score": r.mirror.median_score,
                "mirror_control_median": r.mirror.control_median,
                "mirror_median_decoder_cos": r.mirror.median_decoder_cos,
                "n_live_a": r.mirror.n_live_a,
                "n_live_b": r.mirror.n_live_b,
                "standard_extreme_frac": r.standard_extreme_frac,
                "fve_a": r.fve_a,
                "fve_b": r.fve_b,
                "l0": r.l0,
                "aux_alpha": args.aux_alpha,
                "n_features": args.features,
                "n_concepts": args.concepts,
                "exclusive_frac": args.exclusive_frac,
                "n_excl": getattr(args, "n_excl", None),
                "drop_offset": getattr(args, "drop_offset", None),
                "regime": getattr(args, "regime", None),
            }
            for r in reports
        ]
        with open(args.json, "w") as fh:
            json.dump(payload, fh, indent=2)
        print(f"  wrote {args.json}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="diffdiff", description="Model diffing harness (Phase 0: null test)"
    )
    sub = parser.add_subparsers(dest="command", required=True)
    null = sub.add_parser(
        "null", help="diff a model against itself and measure what is invented"
    )
    _add_null_args(null)
    null.set_defaults(func=_run_regime)

    control = sub.add_parser(
        "control",
        help="positive control: a regime where exclusive concepts genuinely exist. "
             "A null result is only meaningful alongside this.",
    )
    _add_null_args(control)
    control.add_argument("--regime", choices=["cross-arch", "dropped"], default="cross-arch",
                         help="'cross-arch' plants exclusive concepts in both models; "
                              "'dropped' destroys known shared concepts in B only")
    control.add_argument("--n-excl", type=int, default=16,
                         help="number of exclusive (or dropped) ground-truth concepts")
    control.add_argument("--noise", type=float, default=0.02,
                         help="quantization-like noise, for the 'dropped' regime")
    control.add_argument("--drop-offset", type=int, default=0,
                         help="index of the first concept destroyed in the 'dropped' "
                              "regime. Concept frequency follows a power law in the "
                              "index, so offset 0 destroys the MOST active concepts and "
                              "yields an optimistic detection limit; a larger offset "
                              "destroys rarer, harder-to-detect ones")
    control.set_defaults(func=_run_regime)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
