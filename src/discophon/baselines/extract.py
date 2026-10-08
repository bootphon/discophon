"""CLI entry-point to extract units or features from the baselines, for all languages and splits."""

import argparse
from pathlib import Path

import joblib

from discophon.baselines.hubert import extract_hubert_continuous_features, extract_hubert_discrete_units
from discophon.baselines.spidr import extract_spidr_continuous_features, extract_spidr_discrete_units
from discophon.languages import all_languages


def layer_and_path(value: str) -> tuple[int, Path]:
    """Parse `LAYER=PATH`."""
    layer, path = value.split("=", 1)
    return int(layer), Path(path)


def cli(argv: list[str] | None = None) -> None:
    """Command-line entry point for the extraction of units or features."""
    parser = argparse.ArgumentParser(
        prog="discophon.baselines.extract",
        description="Extract discrete units or continuous features from HuBERT or SpidR, "
        "for all languages and splits. Can be resumed if interrupted.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("architecture", type=str, choices=["hubert", "spidr"], help="Model architecture")
    parser.add_argument("kind", type=str, choices=["units", "features"], help="Discrete units or continuous features")
    parser.add_argument("dataset", type=Path, help="Path to the DiscoPhon dataset")
    parser.add_argument("output", type=Path, help="Output directory, with one subdirectory per layer")
    parser.add_argument("checkpoint", type=str, help="Path to the checkpoint (or HuggingFace model for HuBERT)")
    codes = [language.iso_639_3 for language in all_languages()]
    parser.add_argument("--languages", nargs="+", choices=codes, default=codes, metavar="LANG", help="Languages")
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["dev", "test", "train-10min", "train-1h", "train-10h"],
        default=["dev", "test"],
        metavar="SPLIT",
        help="Splits, among dev, test, train-10min, train-1h, and train-10h",
    )
    parser.add_argument("--layers", type=int, nargs="+", help="Layers to extract (all available if not set)")
    parser.add_argument(
        "--kmeans",
        type=layer_and_path,
        action="append",
        metavar="LAYER=PATH",
        help="K-means of a layer, saved with joblib. Required for HuBERT units, and can be repeated",
    )
    parser.add_argument("--batch-size", type=int, default=1, help="Number of utterances per batch (SpidR only)")
    args = parser.parse_args(argv)
    if args.architecture == "hubert" and args.kind == "units" and not args.kmeans:
        parser.error("HuBERT units require at least one `--kmeans LAYER=PATH`.")
    if args.kmeans and (args.architecture, args.kind) != ("hubert", "units"):
        parser.error("`--kmeans` only applies to HuBERT units.")
    if args.architecture == "hubert" and args.batch_size != 1:
        parser.error("HuBERT is not batch invariant: `--batch-size` must be 1.")

    kmeans_by_layer = {layer: joblib.load(path) for layer, path in args.kmeans or []}
    for language in args.languages:
        for split in args.splits:
            match args.architecture, args.kind:
                case "hubert", "units":
                    extract_hubert_discrete_units(
                        args.dataset,
                        args.output,
                        language,
                        split,
                        args.checkpoint,
                        kmeans_by_layer,
                        layers=args.layers,
                    )
                case "hubert", "features":
                    extract_hubert_continuous_features(
                        args.dataset, args.output, language, split, args.checkpoint, layers=args.layers
                    )
                case "spidr", "units":
                    extract_spidr_discrete_units(
                        args.dataset,
                        args.output,
                        language,
                        split,
                        args.checkpoint,
                        layers=args.layers,
                        batch_size=args.batch_size,
                    )
                case "spidr", "features":
                    extract_spidr_continuous_features(
                        args.dataset,
                        args.output,
                        language,
                        split,
                        args.checkpoint,
                        layers=args.layers,
                        batch_size=args.batch_size,
                    )


if __name__ == "__main__":
    cli()
