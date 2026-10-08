"""CLI entry-point for phoneme discovery evaluation."""

import argparse
import json
from pathlib import Path

from discophon.data import DEFAULT_N_UNITS, STEP_UNITS, read_gold_annotations, read_submitted_units
from discophon.evaluate.discovery import phoneme_discovery
from discophon.languages import get_language
from discophon.validate import infer_number_of_phonemes


def cli(argv: list[str] | None = None) -> None:
    """Command-line entry point for phoneme discovery evaluation."""
    parser = argparse.ArgumentParser(
        prog="discophon.evaluate",
        description="Evaluate predicted units on phoneme discovery",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("units", type=Path, help="Path to predicted units")
    parser.add_argument("phones", type=Path, help="Path to gold alignments")
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--language", type=get_language, help="Evaluated language. Either use this or `--n-phonemes`")
    target.add_argument("--n-phonemes", type=int, help="Number of phonemes. Either use this or `--language`")
    parser.add_argument(
        "--n-units",
        type=int,
        help=f"Number of units. Defaults to {DEFAULT_N_UNITS} for many-to-one, and to the number of phonemes "
        "plus one for one-to-one",
    )
    parser.add_argument(
        "--kind",
        type=str,
        choices=["many-to-one", "one-to-one"],
        default="many-to-one",
        help="Kind of assignment (either many-to-one, or one-to-one)",
    )
    parser.add_argument("--step-units", type=int, default=STEP_UNITS, help="Step between units (in ms)")
    args = parser.parse_args(argv)
    if args.n_units is None and args.kind == "many-to-one":
        args.n_units = DEFAULT_N_UNITS
    elif args.n_units is None:
        args.n_units = infer_number_of_phonemes(args.n_phonemes, args.language) + 1
    print(
        json.dumps(
            phoneme_discovery(
                read_submitted_units(args.units),
                read_gold_annotations(args.phones),
                n_units=args.n_units,
                n_phonemes=args.n_phonemes,
                step_units=args.step_units,
                kind=args.kind,
                language=args.language,
            )
        )
    )


if __name__ == "__main__":
    cli()
