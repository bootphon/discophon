"""CLI entry-point for phoneme discovery evaluation."""

import argparse
from pathlib import Path

from discophon.data import read_gold_annotations, read_submitted_units
from discophon.evaluate.discovery import phoneme_discovery
from discophon.languages import get_language

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="discophon.evaluate",
        description="Evaluate predicted units on phoneme discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("units", type=Path, help="Path to predicted units")
    parser.add_argument("phones", type=Path, help="Path to gold alignments")
    parser.add_argument("--language", type=str, help="Evaluated language. Either use this or `--n-phonemes`")
    parser.add_argument("--n-phonemes", type=int, help="Number of phonemes. Either use this or `--language`")
    parser.add_argument("--n-units", type=int, required=True, help="Required. Number of units")
    parser.add_argument(
        "--kind",
        type=str,
        choices=["many-to-one", "one-to-one"],
        default="many-to-one",
        help="Kind of assignment (either many-to-one, or one-to-one)",
    )
    parser.add_argument("--step-units", type=int, default=20, help="Step between units (in ms)")
    args = parser.parse_args()
    if args.n_phonemes is not None:
        n_phonemes = args.n_phonemes
    elif args.language is not None:
        n_phonemes = get_language(args.language).n_phonemes
    else:
        parser.error("Either specify `--language` or `--n-phonemes` in order to specify the number of target phonemes")
    print(
        phoneme_discovery(
            read_submitted_units(args.units),
            read_gold_annotations(args.phones),
            n_units=args.n_units,
            n_phonemes=n_phonemes,
            step_units=args.step_units,
            kind=args.kind,
        )
    )
