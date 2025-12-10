# %%
import polars as pl
from discophon.core.builder import (
    build_items,
    check_consecutive,
    merge_consecutive_duplicate_phones,
    pad_decimal_zeros,
    remove_files_with_spn,
)
from discophon.core.data import read_gold_annotations_as_dataframe

src = "/store/projects/zerospeech/archive/2017/archives/LANG1/output_corpus_creation_wolof/models/alignment_phone/alignment.txt"
dest = "/store/projects/phoneme_discovery/benchmark/alignment/alignment-wol-test.txt"
selection = [
    "WOL_03",
    "WOL_05",
    "WOL_06",
    "WOL_10",
    # "WOL_14",
    # "WOL_16",
    # "WOL_17",
    # "WOL_18",
]

df = (
    pl.read_csv(
        src,
        has_header=False,
        separator=" ",
        new_columns=["#file", "onset", "offset", "proba", "#phone"],
        schema_overrides={
            "#file": pl.String,
            "onset": pl.String,
            "offset": pl.String,
            "proba": pl.Float64,
            "#phone": pl.String,
        },
    )
    .drop("proba")
    .with_columns(pad_decimal_zeros("onset", 6), pad_decimal_zeros("offset", 6))
    .filter(pl.any_horizontal([pl.col("#file").str.starts_with(prefix) for prefix in selection]))
)
df = remove_files_with_spn(merge_consecutive_duplicate_phones(df))
check_consecutive(df)
df.write_cs@(dest, separator=" ")
df = df.with_columns(("WOL_" + pl.col("#file").str.split("_").list.get(1)).alias("speaker"))

# %%
phoneme, triphone = build_items(df)
phoneme.write_csv("/store/projects/phoneme_discovery/benchmark/item/phoneme-wolof.item", separator=" ")
triphone.write_csv("/store/projects/phoneme_discovery/benchmark/item/triphone-wolof.item", separator=" ")

# %%
x = read_gold_annotations_as_dataframe(dest)
y = x.with_columns(pl.col("onset").shift(1).over("file").alias("prev_onset")).filter(pl.col("prev_onset").is_null())
y.with_columns(pl.col("onset").cast(pl.String) == "0.012500")["onset"].all()
# %%
y.filter(pl.col("phone") != "SIL")

# %%
final["phone"].unique()
# %%
final
# %%
x
# %%
df
# %%
df.with_columns(
    pl.col("phone").shift(-1).over("#file").alias("next-phone"),
    pl.col("phone").shift(1).over("#file").alias("prev-phone"),
).drop_nulls().filter(
    pl.col("phone") != "SIL",
    pl.col("next-phone") != "SIL",
    pl.col("prev-phone") != "SIL",
)
# %%
