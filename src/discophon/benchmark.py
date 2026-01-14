from pathlib import Path
from typing import Literal

import polars as pl

from discophon.core import read_gold_annotations, read_submitted_units
from discophon.core.languages import languages_in_split
from discophon.core.validation import validate_dataset_structure, validate_features_structure, validate_units_structure
from discophon.evaluate import phoneme_discovery


def benchmark_discovery(
    path_dataset: str | Path,
    path_units: str | Path,
    *,
    languages: Literal["dev", "test"],
    split: Literal["dev", "test"],
    n_units: int,
    step_units: int,
) -> pl.DataFrame:
    validate_dataset_structure(path_dataset)
    validate_units_structure(path_units, languages=languages, split=split)

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
    from discophon.evaluate.abx import discrete_abx

    validate_dataset_structure(path_dataset)
    validate_units_structure(path_units, languages=languages, split=split)

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
    from discophon.evaluate.abx import continuous_abx

    validate_dataset_structure(path_dataset)
    validate_features_structure(path_features, languages=languages, split=split)

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
        case "abx-discrete":
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
