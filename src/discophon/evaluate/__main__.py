"""CLI entry-point for phoneme discovery evaluation."""

from pathlib import Path

from discophon.data import read_gold_annotations, read_submitted_units
from discophon.evaluate.discovery import phoneme_discovery

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate predicted units on phoneme discovery")
    parser.add_argument("units", type=Path, help="Path to predicted units")
    parser.add_argument("phones", type=Path, help="Path to gold alignments")
    parser.add_argument("--n-phonemes", type=int, required=True, help="Number of phonemes")
    parser.add_argument("--n-units", type=int, required=True, help="Number of units")
    parser.add_argument(
        "--kind",
        type=str,
        choices=["many-to-one", "one-to-one"],
        default="many-to-one",
        help="Kind of assignment (either many-to-one, or one-to-one)",
    )
    parser.add_argument("--step-units", type=int, default=20, help="Step between units (in ms)")
    args = parser.parse_args()
    print(
        phoneme_discovery(
            read_submitted_units(args.units),
            read_gold_annotations(args.phones),
            n_units=args.n_units,
            n_phonemes=args.n_phonemes,
            step_units=args.step_units,
            kind=args.kind,
        )
    )
