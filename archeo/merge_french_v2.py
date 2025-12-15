# %%
from decimal import Decimal

import polars as pl
from discophon.core.builder import pad_decimal_zeros
from segmentation.process import post_process
from segmentation.rttm import rttm_to_annotations

full = pl.read_csv("./french/new_alignment.csv", schema_overrides={"onset": pl.String, "offset": pl.String})
full = full.with_columns(
    full["onset"].str.to_decimal(inference_length=len(full)),
    full["offset"].str.to_decimal(inference_length=len(full)),
)

for speaker in [
    "F01_N",
    "F02_N",
    "F03_N",
    "F04_N",
    "F05_N",
    "F06_N",
    "F07_N",
    "F08_N",
    "F09_N",
    "F10_N",
    "M01_N",
    "M02_N",
    "M03_N",
    "M04_N",
    "M05_N",
    "M06_N",
    "M07_N",
    "M08_N",
    "M09_N",
    "M10_N",
]:
    this_speaker = (
        full.with_columns(
            before=(pl.col("onset").shift(-1).over("file") != pl.col("offset").over("file")).fill_null(value=False),
            after=(pl.col("offset").shift(1).over("file") != pl.col("onset").over("file")).fill_null(value=False),
        )
        .with_columns(pl.col("before").cum_sum().over("file").alias("break"))
        .filter(pl.col("file").str.starts_with(speaker.removesuffix("_N") + "_"))
        .partition_by("break")
    )
    start_idx = 0

    for break_idx, df in enumerate(this_speaker):
        assert df["file"].unique().len() == 1
        filename = str(df["file"][0])

        rttm = (
            df.filter(pl.col("phone") == "SIL")
            .with_columns(start=pl.col("offset"), end=pl.col("onset").shift(-1))
            .with_columns(duration=pl.col("end") - pl.col("start"))
            .drop_nulls()
            .select("start", "duration")
            .with_columns(
                pl.lit("SPEAKER").alias("Type"),
                pl.lit(filename).alias("File ID"),
                pl.col("start").alias("Turn Onset"),
                pl.col("duration").alias("Turn Duration"),
                pl.lit("<NA>").alias("Orthography Field"),
                pl.lit("<NA>").alias("Speaker Type"),
                pl.lit("SPEECH").alias("Speaker Name"),
                pl.lit("<NA>").alias("Confidence Score"),
                pl.lit("<NA>").alias("Signal Lookahead Time"),
            )
            .drop("start", "duration")
            .with_columns(pl.col("Turn Onset").cast(pl.Float64), pl.col("Turn Duration").cast(pl.Float64))
        )
        if rttm.height == 0:
            continue
        annotations = post_process(
            rttm_to_annotations(rttm)[0], min_duration_on=5, min_duration_off=1.0, max_duration_on=30.0
        )
        start_idx += len(annotations)
        if len(annotations) == 0:
            continue
        alignment = (
            pl.concat(
                [
                    (
                        df.filter(pl.col("onset") >= segment.start, pl.col("offset") <= segment.end)
                        .with_columns(
                            pl.lit(f"{speaker}_{i + start_idx:03d}").alias("#file"),
                            pl.col("onset") - pl.lit(Decimal(str(segment.start))),
                            pl.col("offset") - pl.lit(Decimal(str(segment.start))),
                        )
                        .select("#file", "onset", "offset", "phone")
                        .rename({"phone": "#phone"})
                    )
                    for i, segment in enumerate(annotations.itersegments())
                ]
            )
            .with_columns(pl.col("onset").cast(pl.String), pl.col("offset").cast(pl.String))
            .with_columns(pad_decimal_zeros("onset", 6), pad_decimal_zeros("offset", 6))
        )
        alignment.write_csv(f"./french/alignment-{speaker}-{break_idx}.csv", separator=" ")
        with open(f"./french/rttm-{speaker}-{break_idx}.rttm", "w") as f:
            annotations.write_rttm(f)

# %%
