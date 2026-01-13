"""CLI entry-point for phoneme discovery evaluation."""

from pathlib import Path

from discophon.core import read_gold_annotations, read_submitted_units

from .evaluate import phoneme_discovery

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate predicted units on phoneme discovery")
    parser.add_argument("units", type=Path, help="path to predicted units")
    parser.add_argument("phones", type=Path, help="path to gold alignments")
    parser.add_argument("--n-phonemes", type=int, required=True, help="number of phonemes")
    parser.add_argument("--n-units", type=int, required=True, help="number of units")
    parser.add_argument("--step-units", type=int, default=20, help="step between units (in ms)")
    args = parser.parse_args()
    print(
        phoneme_discovery(
            read_submitted_units(args.units),
            read_gold_annotations(args.phones),
            n_units=args.n_units,
            n_phonemes=args.n_phonemes,
            step_units=args.step_units,
        )
    )
