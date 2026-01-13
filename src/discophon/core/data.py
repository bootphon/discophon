from pathlib import Path

import polars as pl
import textgrids

type Units = dict[str, list[int]]
type Phones = dict[str, list[str]]

SAMPLE_RATE = 16_000
FILE, ONSET, OFFSET, PHONE, UNITS = "#file", "onset", "offset", "#phone", "units"


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
    if Path(path).is_file():
        return _read_single_textgrid(path)
    if Path(path).is_dir():
        textgrids = [_read_single_textgrid(p) for p in Path(path).glob("*.TextGrid")]
        return {name: pl.concat(textgrid[name] for textgrid in textgrids).sort("fileid") for name in textgrids[0]}
    raise ValueError(path)


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


def read_gold_annotations_as_dataframe(source: str | Path, *, step_in_ms: int = 10) -> pl.DataFrame:
    phones_per_seconds = 1000 // step_in_ms
    assert step_in_ms * phones_per_seconds == 1000
    df = pl.read_csv(source, separator=" ", columns=[FILE, ONSET, OFFSET, PHONE], schema_overrides=[pl.String] * 4)
    df = df.with_columns(
        df[ONSET].str.to_decimal(inference_length=len(df)),
        df[OFFSET].str.to_decimal(inference_length=len(df)),
    ).sort(FILE, ONSET)
    assert num_invalid_rows(df, step_in_ms=step_in_ms) == 0
    df = df.with_columns(num=(pl.col(OFFSET) - pl.col(ONSET)) * phones_per_seconds)
    assert decimal_series_is_integer(df["num"])
    return df.with_columns(pl.col("num").cast(pl.Int64))


def read_gold_annotations(source: str | Path, *, step_in_ms: int = 10) -> Phones:
    """Read the gold annotations and returns a mapping between file names to the list of phonemes.

    There will be one phoneme each `step_in_ms` millisecond.
    """
    return {
        audio: row[PHONE]
        for audio, row in (
            read_gold_annotations_as_dataframe(source, step_in_ms=step_in_ms)
            .with_columns(pl.col(PHONE).repeat_by("num"))
            .group_by(FILE, maintain_order=True)
            .agg(pl.col(PHONE).explode())
            .rows_by_key(FILE, named=True, unique=True)
            .items()
        )
    }


def read_submitted_units(source: str | Path) -> Units:
    """Read the units from a JSONL file. Must only have fields named 'file' (str) and 'units' (list[int])."""
    return {
        audio: row[UNITS]
        for audio, row in (
            pl.read_ndjson(source, schema_overrides={"file": pl.String, UNITS: pl.List(pl.Int32)})
            .rename({"file": FILE})
            .rows_by_key(FILE, named=True, unique=True)
            .items()
        )
    }
