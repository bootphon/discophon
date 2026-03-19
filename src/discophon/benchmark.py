from pathlib import Path
from typing import Literal

import polars as pl

from discophon.data import DEFAULT_N_UNITS, read_gold_annotations, read_submitted_units
from discophon.evaluate import phoneme_discovery
from discophon.evaluate.assignment import AssignmentKind
from discophon.languages import Language, get_language
from discophon.validate import validate_dataset_structure


def available_languages_and_splits_for_units(path_units: str | Path) -> list[tuple[Language, str]]:
    found = [n.split("-") for n in sorted(p.stem for p in Path(path_units).glob("units-*.jsonl"))]
    return [(get_language(p), "-".join(q)) for _, p, *q in found]


def available_languages_and_splits_for_features(path_features: str | Path) -> list[tuple[Language, str]]:
    return [(get_language(p.parent.stem), p.stem) for p in sorted(Path(path_features).glob("*/*/"))]


def benchmark_discovery(
    path_dataset: str | Path,
    path_units: str | Path,
    *,
    step_units: int,
    kind: AssignmentKind,
) -> pl.DataFrame:
    validate_dataset_structure(path_dataset)
    df = []
    for language, split in available_languages_and_splits_for_units(path_units):
        if split not in {"dev", "test"}:
            continue
        units = read_submitted_units(Path(path_units) / f"units-{language.iso_639_3}-{split}.jsonl")
        phones = read_gold_annotations(Path(path_dataset) / f"alignment/alignment-{language.iso_639_3}-{split}.txt")
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
    step_units: int,
    kind: Literal["triphone", "phoneme"] = "triphone",
) -> pl.DataFrame:
    from discophon.evaluate.abx import discrete_abx

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
    from discophon.evaluate.abx import continuous_abx

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
        "--kind",
        type=str,
        default="many-to-one",
        choices=["many-to-one", "one-to-one"],
        help="Kind of assignment (either many-to-one, or one-to-one)",
    )
    parser.add_argument(
        "--step-units",
        type=int,
        default=20,
        help="Step in ms between units or features (default: 20ms). 'frequency' is then set to 1000 // step_units.",
    )
    args = parser.parse_args()

    match args.benchmark:
        case "discovery":
            out = benchmark_discovery(
                args.dataset,
                args.predictions,
                step_units=args.step_units,
                kind=args.kind,
            )
        case "abx-discrete":
            out = benchmark_abx_discrete(args.dataset, args.predictions, step_units=args.step_units, kind="triphone")
        case "abx-continuous":
            out = benchmark_abx_continuous(args.dataset, args.predictions, step_units=args.step_units, kind="triphone")
        case _:
            parser.error(f"Invalid benchmark: '{args.benchmark}'")
    lock = FileLock(f"{args.output}.lock")
    with lock, args.output.open("a") as f:
        out.write_ndjson(f)
