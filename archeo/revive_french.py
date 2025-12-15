# %%
from decimal import Decimal
from pathlib import Path

import altair as alt
import polars as pl


def rename_speaker(df: pl.DataFrame, col: str) -> pl.DataFrame:
    mapping = {
        "F1": "F01",
        "F2": "F02",
        "F3": "F03",
        "F4": "F04",
        "F5": "F05",
        "F6": "F06",
        "F7": "F07",
        "F8": "F08",
        "F9": "F09",
        "F10": "F10",
        "M1": "M01",
        "M2": "M02",
        "M3": "M03",
        "M4": "M04",
        "M5": "M05",
        "M6": "M06",
        "M7": "M07",
        "M8": "M08",
        "M9": "M09",
        "M10": "M10",
    }
    return (
        df.with_columns(pl.col(col).str.split_exact("_", 0).struct.rename_fields(["speaker"]).struct.unnest())
        .with_columns(pl.col("speaker").replace_strict(mapping).alias("new-speaker"))
        .with_columns(pl.concat_str("new-speaker", pl.col(col).str.strip_prefix(pl.col("speaker"))).alias(col))
        .drop("speaker", "new-speaker")
    )


phones = (
    pl.scan_csv(
        "/home/mpoli/archeo/french/alignment.txt",
        has_header=False,
        new_columns=["segment", "onset", "offset", "proba", "phone"],
        separator=" ",
        schema_overrides={"onset": pl.String, "offset": pl.String},
    )
    .with_columns(
        pl.col("segment")
        .str.split_exact("_", 6)
        .struct.rename_fields(["speaker_id", "book", "chapter", "type", "split", "lang", "segment_idx"])
        .struct.unnest(),
    )
    .with_columns(
        pl.concat_str("speaker_id", pl.lit("_"), "type").alias("speaker"),
        pl.col("segment").str.strip_suffix(pl.concat_str(pl.lit("_"), "segment_idx")).alias("file"),
        pl.col("segment_idx").cast(pl.Int32),
    )
    .filter(pl.col("split") == "te", pl.col("type") == "N")
    .select("segment", "file", "speaker", "segment_idx", "onset", "offset", "phone")
    .sort("segment", "segment", "onset")
    .collect()
)
phones = phones.with_columns(
    phones["onset"].str.to_decimal(inference_length=len(phones)),
    phones["offset"].str.to_decimal(inference_length=len(phones)),
)

segments = (
    pl.concat(
        [
            pl.scan_csv(
                p,
                has_header=False,
                new_columns=["segment", "file", "start", "end"],
                separator=" ",
                schema_overrides={"start": pl.String, "end": pl.String},
            )
            for p in Path("/home/mpoli/archeo/french/seg/").glob("*.txt")
        ]
    )
    .with_columns(
        pl.col("end").cast(pl.Decimal(precision=8, scale=2)),  # Scale in 10ms.
        pl.col("start").cast(pl.Decimal(precision=8, scale=2)),
    )
    .with_columns(duration=pl.col("end") - pl.col("start"))
    .collect()
)
segments = rename_speaker(rename_speaker(segments, "file"), "segment")
segments = segments.filter(pl.col("segment").is_in(phones["segment"].unique().implode()))

full = (
    phones.join(segments, on=["segment", "file"])
    .with_columns(
        onset=pl.col("onset") + pl.col("start") - pl.lit(Decimal("0.0125")),
        offset=pl.col("offset") + pl.col("start") - pl.lit(Decimal("0.0125")),
    )
    .sort("file", "onset")
    .with_columns(
        boundary_next=(pl.col("segment_idx") == (pl.col("segment_idx").shift(1) + 1).over("file")).fill_null(
            value=False
        ),
        boundary_prev=(pl.col("segment_idx") + 1 == pl.col("segment_idx").shift(-1).over("file")).fill_null(
            value=False
        ),
    )
)

# Compute differences between boundaries
boundaries_fixed = (
    full.with_row_index()
    .with_columns(
        diff_next=pl.when(pl.col("boundary_next")).then(pl.col("onset") - pl.col("offset").shift(1)).otherwise(None),
        diff_prev=pl.when(pl.col("boundary_prev")).then(pl.col("onset").shift(-1) - pl.col("offset")).otherwise(None),
    )
    .with_columns(
        onset=pl.when(pl.col("diff_next").is_null())
        .then("onset")
        .otherwise(
            pl.when(pl.col("diff_next") >= Decimal("0.02"))
            .then(pl.col("onset") - Decimal("0.01"))
            .otherwise(pl.col("onset")),
        ),
        offset=pl.when(pl.col("diff_prev").is_null())
        .then("offset")
        .otherwise(
            pl.when((pl.col("diff_prev") == Decimal("0.01")) | (pl.col("diff_prev") == Decimal("0.02")))
            .then(pl.col("offset") + Decimal("0.01"))
            .otherwise(pl.col("offset") + Decimal("0.02"))
        ),
    )
    .with_columns(
        new_diff_next=pl.when(pl.col("boundary_next"))
        .then(pl.col("onset") - pl.col("offset").shift(1))
        .otherwise(None),
        new_diff_prev=pl.when(pl.col("boundary_prev"))
        .then(pl.col("onset").shift(-1) - pl.col("offset"))
        .otherwise(None),
    )
)

assert boundaries_fixed["new_diff_prev"].drop_nulls().sum() == 0
assert boundaries_fixed["new_diff_next"].drop_nulls().sum() == 0
merged = (
    boundaries_fixed.with_columns(
        pl.when(
            (pl.col("phone") == "SIL")
            & (pl.col("file") == pl.col("file").shift(1))
            & (pl.col("phone") == pl.col("phone").shift(1))
        )
        .then(pl.lit(value=False))
        .otherwise(pl.lit(value=True))
        .alias("duplicated_silence"),
    )
    .with_columns(pl.col("duplicated_silence").cum_sum().alias("segment_id"))
    .group_by("file", "phone", "segment_id", maintain_order=True)
    .agg(pl.col("segment").first(), pl.col("onset").first(), pl.col("offset").last(), pl.col("speaker").first())
    .select("segment", "file", "speaker", "onset", "offset", "phone")
)

# Find remaining breaks
breaks = (
    merged.with_columns(
        before=(pl.col("onset").shift(-1).over("file") != pl.col("offset").over("file")).fill_null(value=False),
        after=(pl.col("offset").shift(1).over("file") != pl.col("onset").over("file")).fill_null(value=False),
    )
    .with_columns(pl.col("before").cum_sum().over("file").alias("break"))
    .group_by("file", "break")
    .agg(pl.min("onset"), pl.max("offset"), pl.col("speaker").first())
    .with_columns(duration=pl.col("offset") - pl.col("onset"))
    .with_columns(pl.concat_str("speaker", pl.lit("-"), "break").alias("uid"))
)
chart = (
    alt.Chart(breaks.with_columns(pl.col("duration").cast(pl.Float64)).drop("onset", "offset"))
    .mark_bar()
    .encode(
        x=alt.X("uid:N", sort="-y"),
        y=alt.Y("duration:Q"),
        tooltip=["uid", "duration"],
        color=alt.Color("speaker:N"),
    )
)
chart.save("breaks.html")
chart
# %%
# merged.write_csv("./french/new_alignment.csv")
# %%
chart

# %%
breaks.group_by("speaker").agg(pl.max("break")).filter(pl.col("break") > 0)
# %%
