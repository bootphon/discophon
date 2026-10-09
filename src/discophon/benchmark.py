"""Run the DiscoPhon benchmark on your predictions.

Compute the scores for all languages and splits for which units or features have been extracted.
"""

import argparse
from decimal import Decimal
from pathlib import Path
from typing import Literal

import polars as pl
from filelock import FileLock

from discophon.data import (
    DEFAULT_N_UNITS,
    STEP_UNITS,
    alignment_filename,
    item_filename,
    read_gold_annotations,
    read_submitted_units,
    units_filename,
)
from discophon.evaluate import phoneme_discovery
from discophon.languages import Language, all_languages, get_language
from discophon.validate import validate_dataset_structure

__all__ = ["NoPredictionsError", "benchmark_abx_continuous", "benchmark_abx_discrete", "benchmark_discovery"]

EVALUATED_SPLITS = {"dev", "test"}


class NoPredictionsError(FileNotFoundError):
    """Raised when there are no units or features to evaluate."""

    def __init__(self, directory: str | Path, pattern: str) -> None:
        super().__init__(
            f"Nothing to evaluate in {directory}: expected {pattern}, with the ISO 639-3 code of a DiscoPhon language "
            "and the dev or test split."
        )


def available_languages_and_splits_for_units(
    path_units: str | Path,
    *,
    prefix: str = "units-",
) -> list[tuple[Language, str]]:
    """List of languages and splits for which units are available."""
    found = [n.split("-") for n in sorted(p.stem for p in Path(path_units).glob(f"{prefix}*.jsonl"))]
    codes = {language.iso_639_3 for language in all_languages()}
    if unknown := ["-".join(parts) + ".jsonl" for parts in found if parts[1] not in codes]:
        raise ValueError(
            f"Unknown language codes in {path_units}: {unknown[:5]}. Name the files {prefix}{{code}}-{{split}}.jsonl, "
            "with the ISO 639-3 code of a DiscoPhon language."
        )
    return [(get_language(p), "-".join(q)) for _, p, *q in found]


def available_languages_and_splits_for_features(path_features: str | Path) -> list[tuple[Language, str]]:
    """List of languages and splits for which features are available."""
    found = sorted(Path(path_features).glob("[!.]*/[!.]*/"))  # Ignore hidden directories
    codes = {language.iso_639_3 for language in all_languages()}
    if unknown := sorted({p.parent.name for p in found if p.parent.name not in codes}):
        raise ValueError(
            f"Unknown language codes in {path_features}: {unknown[:5]}. Name the directories {{code}}/{{split}}/, "
            "with the ISO 639-3 code of a DiscoPhon language."
        )
    return [(get_language(p.parent.name), p.name) for p in found]


def benchmark_discovery(
    path_dataset: str | Path,
    path_units: str | Path,
    *,
    kind: Literal["many-to-one", "one-to-one"],
    step_units: int = STEP_UNITS,
) -> pl.DataFrame:
    """Benchmark phoneme discovery. Evaluate all languages and splits with available units.

    The units should be saved in the directory `path_units`, in JSONL files
    named `units-{code}-{split}.jsonl` with keys `file` ([`str`][]) and `units` (`list[int]`).

    Args:
        path_dataset: Path to the DiscoPhon dataset
        path_units: Path to the directory with the predicted units
        kind: Kind of assignment. If it is `many-to-one`, the number of units is set to the default
            ([`DEFAULT_N_UNITS`][discophon.data.DEFAULT_N_UNITS]).
            Otherwise, it is set to the number of phonemes plus one.
        step_units: Step between consecutive units (in ms).

    Returns:
        DataFrame with the results

    Raises:
        NoPredictionsError: If there is nothing to evaluate.

    """
    validate_dataset_structure(path_dataset)
    available = [
        (lang, split)
        for lang, split in available_languages_and_splits_for_units(path_units)
        if split in EVALUATED_SPLITS
    ]
    if not available:
        raise NoPredictionsError(path_units, "units-{code}-{split}.jsonl")
    df = []
    for language, split in available:
        units = read_submitted_units(Path(path_units) / units_filename(language, split))
        phones = read_gold_annotations(Path(path_dataset) / "alignment" / alignment_filename(language, split))
        n_units = DEFAULT_N_UNITS if kind == "many-to-one" else language.n_phonemes + 1
        scores = phoneme_discovery(
            units,
            phones,
            n_units=n_units,
            n_phonemes=language.n_phonemes,
            step_units=step_units,
            kind=kind,
        )
        df.append({"language": language.iso_639_3, "split": split} | scores)
    return pl.DataFrame(df).unpivot(index=["language", "split"], variable_name="metric", value_name="score")


def benchmark_abx_discrete(
    path_dataset: str | Path,
    path_units: str | Path,
    *,
    kind: Literal["triphone", "phoneme"] = "triphone",
    step_units: int = STEP_UNITS,
) -> pl.DataFrame:
    """ABX on all discrete units available.

    The units should be saved in the directory `path_units`, in JSONL files
    named `units-{code}-{split}.jsonl` with keys `file` ([`str`][]) and `units` (`list[int]`).

    Args:
        path_dataset: Path to the DiscoPhon dataset
        path_units: Path to the directory with the predicted units
        kind: Kind of representations to use for ABX computation.
        step_units: Step between consecutive units (in ms).

    Returns:
        DataFrame with the results

    Raises:
        NoPredictionsError: If there is nothing to evaluate.

    """
    from discophon.abx import discrete_abx

    validate_dataset_structure(path_dataset)
    available = [
        (lang, split)
        for lang, split in available_languages_and_splits_for_units(path_units)
        if split in EVALUATED_SPLITS
    ]
    if not available:
        raise NoPredictionsError(path_units, "units-{code}-{split}.jsonl")
    df = []
    for language, split in available:
        abx = discrete_abx(
            Path(path_dataset) / "item" / item_filename(language, split, kind=kind),
            Path(path_units) / units_filename(language, split),
            frequency=Decimal(1_000) / step_units,
            kind=kind,
        )
        for speaker, score in abx.items():
            metric = f"{kind}_abx_discrete_{speaker}"
            df.append({"language": language.iso_639_3, "split": split, "metric": metric, "score": score})
    return pl.DataFrame(df)


def benchmark_abx_continuous(
    path_dataset: str | Path,
    path_features: str | Path,
    *,
    kind: Literal["triphone", "phoneme"] = "triphone",
    step_units: int = STEP_UNITS,
) -> pl.DataFrame:
    """ABX on all continuous features available.

    The features should be saved in the directory `path_features`, in subfolders `path_features/{code}/{split}`.

    Args:
        path_dataset: Path to the DiscoPhon dataset
        path_features: Path to the directory with the extracted features
        kind: Kind of representations to use for ABX computation.
        step_units: Step between consecutive features (in ms).
            The feature frequency will be set to `1_000 / step_units`

    Returns:
        DataFrame with the results

    Raises:
        NoPredictionsError: If there is nothing to evaluate.

    """
    from discophon.abx import continuous_abx

    validate_dataset_structure(path_dataset)
    available = [
        (lang, split)
        for lang, split in available_languages_and_splits_for_features(path_features)
        if split in EVALUATED_SPLITS
    ]
    if not available:
        raise NoPredictionsError(path_features, "{code}/{split}/")
    df = []
    for language, split in available:
        abx = continuous_abx(
            Path(path_dataset) / "item" / item_filename(language, split, kind=kind),
            Path(path_features) / language.iso_639_3 / split,
            frequency=Decimal(1_000) / step_units,
            kind=kind,
        )
        for speaker_context, score in abx.items():
            metric = f"{kind}_abx_continuous_{speaker_context}"
            df.append({"language": language.iso_639_3, "split": split, "metric": metric, "score": score})
    return pl.DataFrame(df)


def cli(argv: list[str] | None = None) -> None:
    """Command-line entry point for the benchmark."""
    parser = argparse.ArgumentParser(
        prog="discophon.benchmark",
        description="Phoneme Discovery benchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("dataset", type=Path, help="Path to the benchmark dataset")
    parser.add_argument("predictions", type=Path, help="Path to the directory with the discrete units or the features")
    parser.add_argument("output", type=Path, help="Path to the output file")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["discovery", "abx-discrete", "abx-continuous"],
        default="discovery",
        help="Which benchmark",
    )
    parser.add_argument(
        "--kind",
        type=str,
        default="many-to-one",
        choices=["many-to-one", "one-to-one"],
        help="Kind of assignment (either many-to-one, or one-to-one). Only applies to '--benchmark discovery'.",
    )
    parser.add_argument(
        "--abx-kind",
        type=str,
        default="triphone",
        choices=["triphone", "phoneme"],
        help="Representation kind for the ABX benchmarks (ignored for '--benchmark discovery').",
    )
    parser.add_argument(
        "--step-units",
        type=int,
        default=STEP_UNITS,
        help="Step in ms between units or features. 'frequency' is then set to 1000 / step_units.",
    )
    args = parser.parse_args(argv)

    match args.benchmark:
        case "discovery":
            out = benchmark_discovery(
                args.dataset,
                args.predictions,
                step_units=args.step_units,
                kind=args.kind,
            )
        case "abx-discrete":
            out = benchmark_abx_discrete(
                args.dataset,
                args.predictions,
                step_units=args.step_units,
                kind=args.abx_kind,
            )
        case "abx-continuous":
            out = benchmark_abx_continuous(
                args.dataset,
                args.predictions,
                step_units=args.step_units,
                kind=args.abx_kind,
            )
        case _:
            parser.error(f"Invalid benchmark: '{args.benchmark}'")
    lock = FileLock(f"{args.output}.lock")
    with lock, args.output.open("a") as f:
        out.write_ndjson(f)


if __name__ == "__main__":
    cli()
