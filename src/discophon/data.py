"""Data loading and writing utilities."""

import itertools
from collections.abc import Iterable
from decimal import Decimal
from pathlib import Path
from typing import Literal, TypedDict

import numpy as np
import polars as pl
import textgrids

__all__ = [
    "DEFAULT_N_UNITS",
    "STEP_PHONES",
    "STEP_UNITS",
    "Phones",
    "Units",
    "read_gold_annotations",
    "read_submitted_units",
]

Splits = Literal["all", "train-10min", "train-1h", "train-10h", "train-100h", "train-all", "dev", "test"]

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


def read_rttm(source: str | Path) -> pl.DataFrame:
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
        textgrids = [_read_single_textgrid(p) for p in Path(path).glob("*.TextGrid")]
        return {name: pl.concat(textgrid[name] for textgrid in textgrids).sort("fileid") for name in textgrids[0]}
    raise ValueError(path)


class TextGridEntry(TypedDict):
    begin: float
    end: float
    label: str


def textgrid_array_from_sequence(seq: Iterable[str | int], *, step_in_ms: int) -> list[TextGridEntry]:
    """Create a list of TextGrid entries from a sequence of tokens."""
    step_in_seconds = Decimal(step_in_ms) / 1000
    labels, counts = zip(*[(key, len(list(group))) for key, group in itertools.groupby(seq)], strict=True)
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
            .then(pl.col(ONSET) != 0)
            .otherwise((pl.col(ONSET) != pl.col(f"prev_{OFFSET}")) | incorrect_duration)
            .alias("valid")
        )["valid"]
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

    There will be one phone every 10 ms.

    Arguments:
        source: Path to the annotations file

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
    """
    return {
        audio: row[UNITS]
        for audio, row in (
            pl.read_ndjson(source, schema_overrides={"file": pl.String, UNITS: pl.List(pl.Int32)})
            .rename({"file": FILE})
            .rows_by_key(FILE, named=True, unique=True)
            .items()
        )
    }
