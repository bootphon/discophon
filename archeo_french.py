# %%
import polars as pl

_ = pl.Config.set_tbl_rows(20)
item = pl.read_csv(
    "/store/scratch/mpoli/data/zerospeech/datasets/zrc2017-test-dataset/french/120s/120s.item",
    separator=" ",
).rename({"#file": "file", "#phone": "phone"})
speakers = item["speaker"].unique().to_list()

align = (
    pl.scan_csv(
        "../archeo/french/alignment.txt",
        has_header=False,
        new_columns=["file", "onset", "offset", "proba", "phone"],
        separator=" ",
    )
    .with_columns(
        pl.col("file")
        .str.split_exact("_", 6)
        .struct.rename_fields(["speaker_id", "book", "chapter", "type", "split", "lang", "segment"])
        .struct.unnest()
    )
    .with_columns(pl.concat_str("speaker_id", pl.lit("_"), "type").alias("speaker"))
    .filter(pl.col("split") == "te", pl.col("type") == "N")
    .drop("proba", "speaker_id", "type")
    .sort("file", "segment", "onset")
    .collect()
)

align
# %%
align["speaker"].unique() == speakers
# %%
len(speakers)
# %%
align["speaker"].unique().sort()
# %%
align["speaker"].value_counts().sort("speaker")
# %%
sorted(speakers)
# %%
align["speaker"].value_counts().sort("speaker")
# %%
