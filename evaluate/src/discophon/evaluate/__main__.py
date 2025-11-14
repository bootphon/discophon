"""CLI entry-point for phoneme discovery evaluation."""

from pathlib import Path

from discophon.core import read_gold_annotations, read_submitted_units

from .evaluate import phoneme_discovery

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("units", type=Path)
    parser.add_argument("phones", type=Path)
    parser.add_argument("--n-units", type=int, default=256)
    parser.add_argument("--n-phones", type=int, default=40)
    parser.add_argument("--step-units", type=int, default=20)
    args = parser.parse_args()
    print(
        phoneme_discovery(
            read_submitted_units(args.units),
            read_gold_annotations(args.phones),
            n_units=args.n_units,
            n_phones=args.n_phones,
            step_units=args.step_units,
            step_phones=args.step_phones,
        )
    )
