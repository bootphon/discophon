"""Data loading and writing utilities."""

import itertools
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import polars as pl
import textgrids

from discophon.languages import Language, all_languages

__all__ = [
    "DEFAULT_N_UNITS",
    "STEP_PHONES",
    "STEP_UNITS",
    "Phones",
    "Splits",
    "Units",
    "alignment_filename",
    "df_to_textgrids",
    "item_filename",
    "manifest_filename",
    "read_gold_annotations",
    "read_rttm",
    "read_scores",
    "read_submitted_units",
    "read_textgrid",
    "rttm_to_textgrids",
    "units_filename",
    "write_textgrids",
]

type Splits = Literal["all", "train-10min", "train-1h", "train-10h", "dev", "test"]
"""Type of the dataset splits. `all` holds every audio file of a language, the other splits link to some of them."""

type Units = dict[str, list[int]]
"""Type of the discrete units: dictionary mapping file identifiers to lists of integers."""

type Phones = dict[str, list[str]]
"""Type of the gold or predicted phones: dictionary mapping file identifiers to list of strings."""

STEP_PHONES = 10
"""Constant step in ms between consecutive phone annotations. Override it in function parameters only if you
use new annotations built differently."""

STEP_UNITS = 20
"""Default step in ms between consecutive units. Corresponds to 50 Hz model. Can be overridden easily."""

DEFAULT_N_UNITS = 256
"""Default number of distinct units in the many-to-one evaluation."""

SAMPLE_RATE = 16_000
FILE, ONSET, OFFSET, PHONE, UNITS = "#file", "onset", "offset", "#phone", "units"


def units_filename(language: Language, split: str) -> str:
    """Filename for the predicted units of a (language, split) pair."""
    return f"units-{language.iso_639_3}-{split}.jsonl"


def alignment_filename(language: Language, split: str) -> str:
    """Filename for the gold phone alignment of a (language, split) pair."""
    return f"alignment-{language.iso_639_3}-{split}.txt"


def item_filename(language: Language, split: str, *, kind: str) -> str:
    """Filename for the ABX item file of a (language, split, kind) triple."""
    return f"{kind}-{language.iso_639_3}-{split}.item"


def manifest_filename(language: Language, split: str) -> str:
    """Filename for the audio manifest of a (language, split) pair."""
    return f"manifest-{language.iso_639_3}-{split}.csv"


def read_rttm(source: str | Path) -> pl.DataFrame:
    """Read an RTTM file, with one column per field of the format."""
    return pl.read_csv(
        source,
        has_header=False,
        new_columns=[
            "Type",
            "File ID",
            "Channel ID",
            "Turn Onset",
            "Turn Duration",
            "Orthography Field",
            "Speaker Type",
            "Speaker Name",
            "Confidence Score",
            "Signal Lookahead Time",
        ],
        separator=" ",
        schema_overrides={
            "Type": pl.String,
            "File ID": pl.String,
            "Turn Onset": pl.Float64,
            "Turn Duration": pl.Float64,
            "Speaker Name": pl.String,
        },
        null_values="<NA>",
    )


def _read_single_textgrid(path: str | Path) -> dict[str, pl.DataFrame]:
    grid = textgrids.TextGrid(path)
    tiers = {}
    for name, tier in grid.items():
        if tier.is_point_tier:
            tiers[name] = pl.DataFrame([{"text": p.text or "SIL", "pos": p.xpos} for p in tier])
        else:
            tiers[name] = pl.DataFrame([{"text": p.text or "SIL", "start": p.xmin, "end": p.xmax} for p in tier])
        tiers[name] = tiers[name].with_columns(fileid=pl.lit(Path(path).stem))
    return tiers


def read_textgrid(path: str | Path) -> dict[str, pl.DataFrame]:
    """Read a TextGrid file or directory of TextGrid files."""
    if Path(path).is_file():
        return _read_single_textgrid(path)
    if Path(path).is_dir():
        grids = [_read_single_textgrid(p) for p in Path(path).glob("*.TextGrid")]
        if not grids:
            raise ValueError(f"No TextGrid files found in directory {path}")
        return {name: pl.concat(grid[name] for grid in grids).sort("fileid") for name in grids[0]}
    raise ValueError(f"Path is neither a TextGrid file nor a directory: {path}")


class TextGridEntry(TypedDict):
    begin: float
    end: float
    label: str


def textgrid_array_from_sequence(seq: Iterable[str | int], *, step_in_ms: int) -> list[TextGridEntry]:
    """Create a list of TextGrid entries from a sequence of tokens."""
    step_in_seconds = Decimal(step_in_ms) / 1000
    groups = [(key, len(list(group))) for key, group in itertools.groupby(seq)]
    if not groups:
        raise ValueError("Cannot build TextGrid intervals from an empty sequence of tokens.")
    labels, counts = zip(*groups, strict=True)
    ends = np.cumsum(counts, dtype=np.int64)
    starts = np.concatenate(([0], ends[:-1]))
    return [
        TextGridEntry(begin=float(starts[i] * step_in_seconds), end=float(ends[i] * step_in_seconds), label=str(label))
        for i, label in enumerate(labels)
    ]


def write_textgrids(seqs: Phones | Units, /, outdir: str | Path, *, tier_name: str, step_in_ms: int) -> None:
    """Write the given sequences of tokens as TextGrid files in the given output directory."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    if empty := sorted(file for file, sequence in seqs.items() if not sequence):
        raise ValueError(f"Cannot write empty sequences to TextGrid: {len(empty)} files, such as {empty[:5]}.")
    for file, sequence in seqs.items():
        path = outdir / f"{file}.TextGrid"
        tg = textgrids.TextGrid(path if path.is_file() else None)
        tg.interval_tier_from_array(tier_name, textgrid_array_from_sequence(sequence, step_in_ms=step_in_ms))
        tg.write(path)


def df_to_textgrids(
    df: pl.DataFrame,
    outdir: str | Path,
    *,
    file_col: str,
    begin_col: str,
    end_col: str,
    label_col: str,
    tier_name: str,
) -> None:
    """Write the intervals of a DataFrame as TextGrid files in `outdir`, one per file and in the tier `tier_name`.

    If a TextGrid file already exists, the tier is added to it.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    for (file,), subdf in df.group_by(file_col, maintain_order=True):
        path = outdir / f"{file}.TextGrid"
        tg = textgrids.TextGrid(path if path.is_file() else None)
        array = [
            TextGridEntry(begin=row[begin_col], end=row[end_col], label=row[label_col])
            for row in subdf.sort(begin_col).iter_rows(named=True)
        ]
        tg.interval_tier_from_array(tier_name, array)
        tg.write(path)


def rttm_to_textgrids(source: str | Path, outdir: str | Path, *, tier_name: str) -> None:
    """Write the speaker turns of an RTTM file as TextGrid files in `outdir`, in the tier `tier_name`."""
    df_to_textgrids(
        read_rttm(source).with_columns((pl.col("Turn Onset") + pl.col("Turn Duration")).alias("Turn Offset")),
        outdir,
        file_col="File ID",
        begin_col="Turn Onset",
        end_col="Turn Offset",
        label_col="Speaker Name",
        tier_name=tier_name,
    )


def num_invalid_rows(df: pl.DataFrame, *, step_in_ms: int) -> int:
    """For each file, the first entry starts at 0 and each subsequent entry starts where the previous has ended."""
    incorrect_duration = ~(step_in_ms / 1000 <= pl.col(OFFSET) - pl.col(ONSET))
    return int(
        df.with_columns(pl.col(OFFSET).shift(1).over(FILE).alias(f"prev_{OFFSET}"))
        .with_columns(
            pl.when(pl.col(f"prev_{OFFSET}").is_null())
            .then((pl.col(ONSET) != 0) | incorrect_duration)
            .otherwise((pl.col(ONSET) != pl.col(f"prev_{OFFSET}")) | incorrect_duration)
            .alias("invalid")
        )["invalid"]
        .sum()
    )


def decimal_series_is_integer(series: pl.Series) -> bool:
    return (
        series.cast(pl.String)
        .str.split_exact(".", 1)
        .struct.rename_fields(["integer", "fractional"])
        .struct.field("fractional")
        .str.replace_all("0", "")
        .eq("")
        .all()
    )


def read_gold_annotations_as_dataframe(source: str | Path) -> pl.DataFrame:
    df = pl.read_csv(source, separator=" ", columns=[FILE, ONSET, OFFSET, PHONE], schema_overrides=[pl.String] * 4)
    return df.with_columns(
        df[ONSET].str.to_decimal(inference_length=len(df)),
        df[OFFSET].str.to_decimal(inference_length=len(df)),
    ).sort(FILE, ONSET)


class AnnotationsError(ValueError):
    def __init__(self, step_in_ms: int) -> None:
        super().__init__(
            "Invalid annotations: each entry should start where the previous one has ended, "
            f"and last at least {step_in_ms} ms."
        )


def read_gold_annotations(source: str | Path, *, step_in_ms: int = STEP_PHONES) -> Phones:
    """Read the gold annotations and return a mapping between file names to the list of phonemes.

    There will be one phone every `step_in_ms` ms.

    Arguments:
        source: Path to the annotations file
        step_in_ms: Step in ms between each phone.

    Returns:
        Mapping between file ids and phones

    """
    phones_per_seconds = 1000 // step_in_ms
    if step_in_ms * phones_per_seconds != 1000:
        raise ValueError(f"step_in_ms={step_in_ms} is not valid, it should be a divisor of 1000.")
    df = read_gold_annotations_as_dataframe(source)
    if num_invalid_rows(df, step_in_ms=step_in_ms) > 0:
        raise AnnotationsError(step_in_ms)
    df = df.with_columns(num=(pl.col(OFFSET) - pl.col(ONSET)) * phones_per_seconds)
    if not decimal_series_is_integer(df["num"]):
        raise ValueError(f"Each phone should last a multiple of {step_in_ms} ms, but found some that don't.")
    return {
        audio: row[PHONE]
        for audio, row in (
            df.with_columns(pl.col("num").cast(pl.Int64))
            .with_columns(pl.col(PHONE).repeat_by("num"))
            .group_by(FILE, maintain_order=True)
            .agg(pl.col(PHONE).explode())
            .rows_by_key(FILE, named=True, unique=True)
            .items()
        )
    }


def read_submitted_units(source: str | Path) -> Units:
    """Read the units from a JSONL file. Must only have fields named `file` ([`str`][]) and `units` (`list[int]`).

    Arguments:
        source: Path to the units file

    Returns:
        Mapping between file ids and units

    Raises:
        ValueError: If a file appears more than once, or if some units are missing.

    """
    df = pl.read_ndjson(source, schema_overrides={"file": pl.String, UNITS: pl.List(pl.Int32)}).rename({"file": FILE})
    if duplicates := sorted(set(df.filter(pl.col(FILE).is_duplicated())[FILE])):
        raise ValueError(f"Duplicate files in {source}: {len(duplicates)} files, such as {duplicates[:5]}.")
    is_null = pl.col(UNITS).is_null() | pl.col(UNITS).list.eval(pl.element().is_null()).list.any()
    if missing := sorted(df.filter(is_null)[FILE]):
        raise ValueError(f"Missing or null units in {source}: {len(missing)} files, such as {missing[:5]}.")
    return {audio: row[UNITS] for audio, row in df.rows_by_key(FILE, named=True, unique=True).items()}


def read_scores(root: str | Path) -> pl.DataFrame:
    """Read the scores of the [artifacts dataset](https://huggingface.co/datasets/coml/discophon-artifacts).

    Reads every `{model}/{condition}/{folder}/{layer}/scores.jsonl` in the dataset, or in one model directory if
    `root` holds an `info.json`. The condition is `zero-shot` or `ft-{language}-{duration}`, and the folder is
    `many_to_one`, `one_to_one`, `continuous`, or `many_to_one-k{N}`.

    ABX scores within and across speakers are also averaged, under the name of the metric without the speaker
    condition. For example, `triphone_abx_continuous` averages `triphone_abx_continuous_within_speaker` and
    `triphone_abx_continuous_across_speaker`, and `phoneme_abx_discrete_any_context` averages
    `phoneme_abx_discrete_within_speaker_any_context` and `phoneme_abx_discrete_across_speaker_any_context`.

    Arguments:
        root: Path to the artifacts dataset, or to one model directory in it

    Returns:
        DataFrame with one row per score and columns `model`, `folder`, `ft_lang` (null for zero-shot),
            `duration` (`"0"` for zero-shot), `layer`, `split`, `language`, `language_split` (whether the
            language is a dev or test language), `metric`, and `score`, as stored (not in %).

    """
    root = Path(root).resolve()  # the model name is read from the path, so `.` must be resolved
    model_dirs = [root] if (root / "info.json").exists() else sorted(p.parent for p in root.glob("*/info.json"))
    paths = [p for model_dir in model_dirs for p in sorted(model_dir.glob("*/*/*/scores.jsonl"))]
    if not paths:
        raise ValueError(f"No scores in {root}.")
    scores = (
        pl.concat(
            [
                pl.read_ndjson(p).with_columns(
                    model=pl.lit(p.parts[-5]),
                    condition=pl.lit(p.parts[-4]),
                    folder=pl.lit(p.parts[-3]),
                    layer=pl.lit(int(p.parts[-2]), dtype=pl.Int64),
                )
                for p in paths
            ]
        )
        .with_columns(
            ft_lang=pl.col("condition").str.extract(r"^ft-([a-z]{3})-"),
            duration=pl.col("condition").str.extract(r"^ft-[a-z]{3}-(.+)$").fill_null("0"),
            language_split=pl.col("language").replace_strict({lang.iso_639_3: lang.split for lang in all_languages()}),
        )
        .select(
            "model", "folder", "ft_lang", "duration", "layer", "split", "language", "language_split", "metric", "score"
        )
    )
    keys = [c for c in scores.columns if c != "score"]
    if not (duplicates := scores.filter(scores.select(keys).is_duplicated())).is_empty():
        raise ValueError(f"Duplicate scores in {root}: {duplicates.row(0, named=True)}.")
    speakers = r"^(.+_abx_(?:discrete|continuous))_(?:within|across)_speaker(.*)$"
    averaged = (
        scores.filter(pl.col("metric").str.contains(speakers))
        .with_columns(pl.col("metric").str.replace(speakers, "${1}${2}"))
        .group_by(keys, maintain_order=True)
        .agg(pl.mean("score"), n=pl.len())
    )
    if not averaged.filter(pl.col("n") != 2).is_empty():
        raise ValueError(f"ABX within speaker without across speaker (or the opposite) in {root}.")
    return pl.concat([scores, averaged.drop("n")]).sort(keys)
