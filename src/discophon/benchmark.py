from itertools import product
from pathlib import Path
from typing import Literal

import polars as pl

from .data import read_gold_annotations, read_submitted_units
from .evaluate import phoneme_discovery
from .languages import Language, dev_languages, get_language, test_languages


class DatasetError(ValueError):
    def __init__(self) -> None:
        super().__init__("Invalid phoneme_discovery dataset structure. Verify your file structure!")


def validate_dataset_structure(path: str | Path) -> None:
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


def available_languages_and_splits_for_units(path_units: str | Path) -> list[tuple[Language, str]]:
    found = [n.split("-") for n in sorted(p.stem for p in Path(path_units).glob("units-*.jsonl"))]
    return [(get_language(p), "-".join(q)) for _, p, *q in found]


def available_languages_and_splits_for_features(path_features: str | Path) -> list[tuple[Language, str]]:
    return [(get_language(p.parent.stem), p.stem) for p in sorted(Path(path_features).glob("*/*/"))]


def benchmark_discovery(
    path_dataset: str | Path,
    path_units: str | Path,
    *,
    n_units: int,
    step_units: int,
) -> pl.DataFrame:
    validate_dataset_structure(path_dataset)
    df = []
    for language, split in available_languages_and_splits_for_units(path_units):
        if split not in {"dev", "test"}:
            continue
        units = read_submitted_units(Path(path_units) / f"units-{language.iso_639_3}-{split}.jsonl")
        phones = read_gold_annotations(Path(path_dataset) / f"alignment/alignment-{language.iso_639_3}-{split}.txt")
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
    step_units: int,
    kind: Literal["triphone", "phoneme"] = "triphone",
) -> pl.DataFrame:
    from .evaluate.abx import discrete_abx

    validate_dataset_structure(path_dataset)
    df = []
    for language, split in available_languages_and_splits_for_units(path_units):
        if split not in {"dev", "test"}:
            continue
        abx = discrete_abx(
            Path(path_dataset) / f"item/{kind}-{language.iso_639_3}-{split}.item",
            Path(path_units) / f"units-{language.iso_639_3}-{split}.jsonl",
            frequency=1_000 // step_units,
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
    step_units: int,
    kind: Literal["triphone", "phoneme"] = "triphone",
) -> pl.DataFrame:
    from .evaluate.abx import continuous_abx

    validate_dataset_structure(path_dataset)
    df = []
    for language, split in available_languages_and_splits_for_features(path_features):
        if split not in {"dev", "test"}:
            continue
        abx = continuous_abx(
            Path(path_dataset) / f"item/{kind}-{language.iso_639_3}-{split}.item",
            Path(path_features) / f"{language.iso_639_3}/{split}",
            frequency=1_000 // step_units,
            kind=kind,
        )
        for speaker_context, score in abx.items():
            metric = f"{kind}_abx_continuous_{speaker_context}"
            df.append({"language": language.iso_639_3, "split": split, "metric": metric, "score": score})
    return pl.DataFrame(df)


if __name__ == "__main__":
    import argparse

    from filelock import FileLock

    parser = argparse.ArgumentParser(description="Phoneme Discovery benchmark")
    parser.add_argument("dataset", type=Path, help="Path to the benchmark dataset")
    parser.add_argument("predictions", type=Path, help="Path to the directory with the discrete units or the features")
    parser.add_argument("output", type=Path, help="Path to the output file")
    parser.add_argument(
        "--benchmark",
        type=str,
        choices=["discovery", "abx-discrete", "abx-continuous"],
        default="discovery",
        help="Which benchmark (default: discovery)",
    )
    parser.add_argument(
        "--n-units",
        type=int,
        help="Number of discrete units. Required if benchmark is 'discovery'",
    )
    parser.add_argument(
        "--step-units",
        type=int,
        default=20,
        help="Step in ms between units or features (default: 20ms). 'frequency' is then set to 1000 // step_units.",
    )
    parser.add_argument(
        "--kind",
        type=str,
        choices=["triphone", "phoneme"],
        default="triphone",
        help="Use either triphone or phoneme representations for ABX discriminability",
    )
    args = parser.parse_args()

    match args.benchmark:
        case "discovery":
            if args.n_units is None:
                parser.error("--n-units must be set if benchmark is 'discovery'")
            out = benchmark_discovery(args.dataset, args.predictions, step_units=args.step_units, n_units=args.n_units)
        case "abx-discrete":
            out = benchmark_abx_discrete(args.dataset, args.predictions, step_units=args.step_units, kind=args.kind)
        case "abx-continuous":
            out = benchmark_abx_continuous(args.dataset, args.predictions, step_units=args.step_units, kind=args.kind)
        case _:
            parser.error(f"Invalid benchmark: '{args.benchmark}'")
    with FileLock(f"{args.output}.lock"), args.output.open("a") as f:
        out.write_ndjson(f)
