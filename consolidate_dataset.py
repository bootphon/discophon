from pathlib import Path

import polars as pl
import soundfile as sf
from tqdm import tqdm

from discophon.data import read_rttm

root = Path("/lustre/fsn1/projects/rech/iyn/uvw27yn/mms_ulab_v2_segmented")

rttm = (
    read_rttm("./mms_ulab_v2.rttm")
    .select("File ID", "Turn Onset", "Turn Duration")
    .rename({"File ID": "orig_id", "Turn Onset": "turn_onset", "Turn Duration": "turn_duration"})
    .with_columns((pl.col("orig_id").cum_count().over("orig_id").alias("count") - 1).cast(pl.String).str.zfill(5))
    .with_columns(pl.concat_str("orig_id", pl.lit("_"), "count").alias("id"))
    .drop("count")
)


df = (
    pl.read_ndjson("./mms_ulab_v2.jsonl")
    .rename({"fileid": "id"})
    .with_columns(
        pl.col("path").str.split("/").list.slice(-2, -1).list.join("/").alias("file_name"),
        pl.col("id").str.split("_").list.get(0).alias("iso3"),
        pl.col("id").str.split("_").list.slice(0, 2).list.join("_").alias("orig_id"),
    )
    .join(rttm, on=["id", "orig_id"])
    .with_columns(no_match=pl.col("num_samples") != (pl.col("turn_duration") * 16_000).cast(pl.Int64))
)

actual = []
for no_match, num_samples, path in tqdm(df[["no_match", "num_samples", "file_name"]].iter_rows(), total=len(df)):
    if no_match:
        actual.append(len(sf.read(root / path, always_2d=True)[0]))
    else:
        actual.append(num_samples)

df = df.with_columns(num_samples=pl.Series(actual))
splits = pl.concat(
    (
        pl.scan_ndjson("./mms_ulab_v2_train.jsonl")
        .select("fileid")
        .rename({"fileid": "id"})
        .with_columns(split=pl.lit("train")),
        pl.scan_ndjson("./mms_ulab_v2_val.jsonl")
        .select("fileid")
        .rename({"fileid": "id"})
        .with_columns(split=pl.lit("val")),
    )
).collect()
df = (
    df.join(splits, on="id", how="full")
    .with_columns(pl.col("split").fill_null("excluded"))
    .sort("id")
    .select("id", "orig_id", "iso3", "file_name", "num_samples", "turn_onset", "turn_duration", "split")
)
df.write_ndjson("/lustre/fsn1/projects/rech/iyn/uvw27yn/mms_ulab_v2_segmented/metadata.jsonl")
