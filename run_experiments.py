"""CLI entry point of the experiment harness (Part A, Phase 2).

Quick check:
    python run_experiments.py --preset quick --out results/phase2_quick.csv --overwrite

See README.md ("Running experiments") and report/mismatch_protocol.md.
"""

import argparse
import sys

from denoisers import METHODS
from experiments.arms import ARMS, VST_INVERSES
from experiments.runner import apply_preset, run_experiments
from utils import load_config


def build_parser():
    p = argparse.ArgumentParser(description="Mismatch-protocol experiment harness.")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--preset", default=None, help="config section overlaid on the defaults, e.g. 'quick'")
    p.add_argument("--datasets", nargs="+", default=None)
    p.add_argument("--methods", nargs="+", default=None, choices=METHODS)
    p.add_argument("--arms", nargs="+", default=None, choices=ARMS)
    p.add_argument("--max-images", type=int, default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--out", default=None, help="results CSV (default: <results_dir>/phase2_results.csv)")
    p.add_argument("--vst-inverse", choices=VST_INVERSES, default=None,
                   help="inverse GAT of the VST arm; anything but closed_form renames the arm VST-<inverse>")
    p.add_argument("--resume", action="store_true", help="skip rows already in the CSV")
    p.add_argument("--overwrite", action="store_true", help="replace an existing CSV instead of refusing")
    p.add_argument("--dry-run", action="store_true", help="print the number of runs and exit")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    cfg = apply_preset(load_config(args.config), args.preset)
    out = args.out or f"{cfg.get('results_dir', 'results')}/phase2_results.csv"
    summary = run_experiments(
        cfg,
        datasets=args.datasets or cfg["datasets"],
        methods=args.methods or list(METHODS),
        arms=args.arms or cfg["arms"],
        out=out,
        max_images=args.max_images if args.max_images is not None else cfg.get("max_images"),
        repeats=args.repeats or cfg.get("repeats", 1),
        seed=args.seed,
        vst_inverse=args.vst_inverse or cfg.get("vst_inverse", "closed_form"),
        resume=args.resume,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
    )
    if not args.dry_run:
        print(f"wrote {summary['written']} rows to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
