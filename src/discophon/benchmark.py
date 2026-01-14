from itertools import product
from pathlib import Path
from typing import Literal

import polars as pl

from .data import read_gold_annotations, read_submitted_units
from .evaluate import phoneme_discovery
from .languages import dev_languages, languages_in_split, test_languages


class DatasetError(ValueError):
    def __init__(self) -> None:
        super().__init__("Invalid phoneme_discovery dataset structure. Verify your file structure!")


def _validate_dataset_structure(path: str | Path) -> None:
    root = Path(path).resolve()
    languages = dev_languages() + test_languages()
    if {p.name for p in root.glob("*")} != {"alignment", "audio", "item", "manifest"}:
        raise DatasetError
    if {p.name for p in (root / "alignment").glob("*")} != {
        f"alignment-{lang.iso_639_3}-{split}.txt" for lang, split in product(languages, ["dev", "test"])
    }:
        raise DatasetError
    if {p.name for p in (root / "item").glob("*")} != {
        f"{kind}-{lang.iso_639_3}-{split}.item"
        for kind, lang, split in product(["triphone", "phoneme"], languages, ["dev", "test"])
    }:
        raise DatasetError
    if {p.name for p in (root / "manifest").glob("*")} != (
        {
            f"manifest-{lang.iso_639_3}-{split}.csv"
            for lang, split in product(languages, ["dev", "test", "train-10h", "train-10min", "train-1h"])
        }
        | {"speakers.jsonl"}
    ):
        raise DatasetError
    if {p.name for p in (root / "audio").glob("*")} != {lang.iso_639_3 for lang in languages}:
        raise DatasetError
    splits = {"all", "dev", "test", "train-10h", "train-10min", "train-1h"}
    for lang in languages:
        if {p.stem for p in (root / "audio" / lang.iso_639_3).glob("*")} != splits:
            raise DatasetError


def _validate_units_structure(
    path: str | Path,
    *,
    languages: Literal["dev", "test"],
    split: Literal["dev", "test"],
) -> None:
    expected = {f"units-{lang.iso_639_3}-{split}.jsonl" for lang in languages_in_split(languages)}
    found = {p.name for p in Path(path).glob("*.jsonl")}
    if not expected.issubset(found):
        raise ValueError(f"Missing units. Expected in {path}:\n{list(expected)}")


def _validate_features_structure(
    path: str | Path,
    *,
    languages: Literal["dev", "test"],
    split: Literal["dev", "test"],
) -> None:
    expected = {f"{lang.iso_639_3}/{split}" for lang in languages_in_split(languages)}
    found = {str(p.relative_to(path)) for p in Path(path).glob("*/*")}
    if not expected.issubset(found):
        raise ValueError(f"Missing directories with features. Expected in {path}:\n{list(expected)}")


def benchmark_discovery(
    path_dataset: str | Path,
    path_units: str | Path,
    *,
    languages: Literal["dev", "test"],
    split: Literal["dev", "test"],
    n_units: int,
    step_units: int,
) -> pl.DataFrame:
    _validate_dataset_structure(path_dataset)
    _validate_units_structure(path_units, languages=languages, split=split)

    df = []
    for language in languages_in_split(languages):
        phones = read_gold_annotations(Path(path_dataset) / f"alignments/alignment-{language.iso_639_3}-{split}.txt")
        units = read_submitted_units(Path(path_units) / f"units-{language.iso_639_3}-{split}.jsonl")
        scores = phoneme_discovery(
            units,
            phones,
            n_units=n_units,
            n_phonemes=language.n_phonemes,
            step_units=step_units,
        )
        df.append({"language": language.iso_639_3, "split": split} | scores)
    return pl.DataFrame(df).unpivot(index=["language", "split"], variable_name="metric", value_name="score")


def benchmark_abx_discrete(
    path_dataset: str | Path,
    path_units: str | Path,
    *,
    languages: Literal["dev", "test"],
    split: Literal["dev", "test"],
    step_units: int,
) -> pl.DataFrame:
    from .evaluate.abx import discrete_abx

    _validate_dataset_structure(path_dataset)
    _validate_units_structure(path_units, languages=languages, split=split)

    df = []
    for language in languages_in_split(languages):
        for kind in ("triphone", "phoneme"):
            abx = discrete_abx(
                Path(path_dataset) / f"item/{kind}-{language.iso_639_3}-{split}.item",
                Path(path_units) / f"units-{language.iso_639_3}-{split}.jsonl",
                frequency=1_000 // step_units,
            )
            for speaker in ("within", "across"):
                metric = f"{kind}_abx_discrete_{speaker}_speaker"
                df.append({"language": language.iso_639_3, "split": split, "metric": metric, "score": abx[speaker]})
    return pl.DataFrame(df)


def benchmark_abx_continuous(
    path_dataset: str | Path,
    path_features: str | Path,
    *,
    languages: Literal["dev", "test"],
    split: Literal["dev", "test"],
    step_units: int,
) -> pl.DataFrame:
    from .evaluate.abx import continuous_abx

    _validate_dataset_structure(path_dataset)
    _validate_features_structure(path_features, languages=languages, split=split)

    df = []
    for language in languages_in_split(languages):
        for kind in ("triphone", "phoneme"):
            abx = continuous_abx(
                Path(path_dataset) / f"item/{kind}-{language.iso_639_3}-{split}.item",
                Path(path_features) / f"{language.iso_639_3}/{split}",
                frequency=1_000 // step_units,
            )
            for speaker in ("within", "across"):
                metric = f"{kind}_abx_continuous_{speaker}_speaker"
                df.append({"language": language.iso_639_3, "split": split, "metric": metric, "score": abx[speaker]})
    return pl.DataFrame(df)


if __name__ == "__main__":
    import argparse

    from filelock import FileLock

    parser = argparse.ArgumentParser(description="Phoneme Discovery benchmark")
    parser.add_argument("dataset", type=Path, help="Path to the benchmark dataset")
    parser.add_argument("predictions", type=Path, help="Path to the directory with the discrete units or the features")
    parser.add_argument("output", type=Path, help="Path to the output file")
    parser.add_argument("--languages", required=True, choices=["dev", "test"], help="Which language split")
    parser.add_argument("--split", required=True, choices=["dev", "test"], help="Which subset")
    parser.add_argument(
        "--benchmark",
        choices=["discovery", "abx-discrete", "abx-continuous"],
        default="discovery",
        help="Which benchmark (default: discovery)",
    )
    parser.add_argument("--n-units", type=int, help="Number of discrete units. Required if benchmark is 'discovery'")
    parser.add_argument(
        "--step-units",
        type=int,
        default=20,
        help="Step in ms between units or features (default: 20ms)",
    )
    args = parser.parse_args()

    match args.benchmark:
        case "discovery":
            fn, kwargs = benchmark_discovery, {"n_units": args.n_units}
        case "abx-discrete":
            fn, kwargs = benchmark_abx_discrete, {}
        case "abx-continuous":
            fn, kwargs = benchmark_abx_continuous, {}
        case _:
            raise ValueError(args.benchmark)
    scores = fn(
        args.dataset,
        args.predictions,
        languages=args.languages,
        split=args.split,
        step_units=args.step_units,
        **kwargs,
    )
    with FileLock(f"{args.output}.lock"), args.output.open("a") as f:
        scores.write_ndjson(f)
