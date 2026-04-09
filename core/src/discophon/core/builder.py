# To move to builder once cleaned up

from pathlib import Path

import polars as pl

IPA_LONG = "ː"


def check_consecutive(df: pl.DataFrame) -> bool:
    contiguous = (
        pl.when(pl.col("#file") == pl.col("#file").shift(1))
        .then(pl.col("onset") == pl.col("offset").shift(1))
        .otherwise(statement=True)
        .alias("contiguous")
    )
    return df.with_columns(contiguous)["contiguous"].all()


def merge_consecutive_duplicate_phones(df: pl.DataFrame, vowels: set[str]) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.when(
                (pl.col("fileid") == pl.col("fileid").shift(1))
                & (pl.col("phone") == pl.col("phone").shift(1))
                & (pl.col("word_index") == pl.col("word_index").shift(1))
            )
            .then(pl.lit(value=False))
            .otherwise(pl.lit(value=True))
            .alias("is_new_segment")
        )
        .with_columns(pl.col("is_new_segment").cum_sum().alias("segment_id"))
        .group_by("fileid", "phone", "segment_id", maintain_order=True)
        .agg(pl.col("onset").first(), pl.col("offset").last(), pl.col("phone").alias("all_phones"))
        .select("fileid", "onset", "offset", "phone", "all_phones")
        .with_columns(
            pl.when(pl.col("all_phones").list.len() > 1, pl.col("all_phones").list.first().is_in(vowels))
            .then(pl.col("all_phones").list.first() + IPA_LONG)
            .otherwise(pl.col("all_phones").list.first())
            .alias("phone")
        )
        .drop("all_phones")
        .sort("fileid")
    )


def remove_files_with_spn(df: pl.DataFrame) -> pl.DataFrame:
    to_remove = (
        df.group_by("#file")
        .agg(pl.col("#phone").unique().alias("unique_phones"))
        .filter(pl.col("unique_phones").list.contains("SPN"))["#file"]
        .to_list()
    )
    return df.filter(~pl.col("#file").is_in(to_remove))


def pad_decimal_zeros(col: str, length: int) -> pl.Expr:
    return (
        pl.col(col)
        .str.split_exact(".", 1)
        .struct.rename_fields(["int_part", "dec_part"])
        .struct.with_fields(
            pl.when(pl.field("dec_part").is_null())
            .then(pl.lit("0").str.pad_end(length, "0"))
            .otherwise(pl.field("dec_part").str.pad_end(length, "0"))
            .alias("dec_part")
        )
        .struct.with_fields((pl.field("int_part") + "." + pl.field("dec_part")).alias(col))
        .struct.field(col)
    )


def read_lexicon(path: str | Path) -> pl.DataFrame:
    return (
        pl.read_csv(path, has_header=False, new_columns=["full"])
        .with_columns(
            pl.col("full").str.split(" ").list.get(0).alias("word"),
            pl.col("full").str.split(" ").list.slice(1).alias("phones"),
        )
        .select("word", "phones")
    )


def read_text(path: str | Path) -> pl.DataFrame:
    return (
        pl.read_csv(path, has_header=False, new_columns=["full"])
        .with_columns(
            pl.col("full").str.split(" ").list.get(0).alias("fileid"),
            pl.col("full").str.split(" ").list.slice(1).alias("words"),
        )
        .select("fileid", "words")
    )


def read_old_alignments(path: str | Path) -> pl.DataFrame:
    return (
        pl.read_csv(
            path,
            has_header=False,
            separator=" ",
            new_columns=["fileid", "onset", "offset", "proba", "phone"],
            schema_overrides={"onset": pl.String, "offset": pl.String},
        )
        .drop("proba")
        .with_columns(pad_decimal_zeros("onset", 6), pad_decimal_zeros("offset", 6))
    )


def build_wolof_alignments(
    lexicon: pl.DataFrame,
    text: pl.DataFrame,
    alignments: pl.DataFrame,
    phonology: dict[str, str],
    may_be_long: set[str],
) -> pl.DataFrame:
    g2p = {k: v for k, (v,) in lexicon.rows_by_key("word", unique=True).items()}
    y = alignments.with_columns(phone_index())
    z = words_to_exploded_phones(text, g2p)
    x = y.join(z, on=["fileid", "phone", "index"], how="left").with_columns(pl.col("phone").replace_strict(phonology))
    return merge_consecutive_duplicate_phones(x, may_be_long)


def build_items(df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    phoneme = (
        df.with_columns(
            pl.col("#phone").shift(-1).over("#file").alias("next-phone"),
            pl.col("#phone").shift(1).over("#file").alias("prev-phone"),
        )
        .drop_nulls()
        .filter(
            pl.col("#phone") != "SIL",
            pl.col("next-phone") != "SIL",
            pl.col("prev-phone") != "SIL",
        )
        .select("#file", "onset", "offset", "#phone", "prev-phone", "next-phone", "speaker")
    )
    triphone = (
        df.with_columns(
            pl.col("#phone").shift(-1).over("#file").alias("next-phone"),
            pl.col("#phone").shift(1).over("#file").alias("prev-phone"),
            pl.col("offset").shift(-1).over("#file").alias("next-offset"),
            pl.col("onset").shift(1).over("#file").alias("prev-onset"),
        )
        .drop_nulls()
        .filter(
            pl.col("#phone") != "SIL",
            pl.col("next-phone") != "SIL",
            pl.col("prev-phone") != "SIL",
        )
        .drop("onset", "offset")
        .rename({"prev-onset": "onset", "next-offset": "offset"})
        .select("#file", "onset", "offset", "#phone", "prev-phone", "next-phone", "speaker")
    )
    return phoneme, triphone


def remove_last_entry_from_item(df: pl.DataFrame) -> pl.DataFrame:
    return (
        df.with_columns(
            pl.when(pl.col("#file") == pl.col("#file").shift(-1))
            .then(pl.lit(True))
            .otherwise(pl.lit(False))
            .alias("not_last")
        )
        .filter(pl.col("not_last"))
        .drop("not_last")
    )


def phone_index() -> pl.Expr:
    return (
        pl.when(pl.col("phone") == "SIL")
        .then(pl.lit(None))
        .otherwise(pl.col("phone").ne("SIL").cast(pl.Int64).cum_sum().over("fileid") - 1)
        .alias("index")
    )


def words_to_exploded_phones(df: pl.DataFrame, g2p: dict[str, list[str]]) -> pl.DataFrame:
    return (
        df.with_columns(pl.col("words").list.eval(pl.element().replace_strict(g2p, default=None)).alias("phones"))
        .with_columns(pl.col("phones").list.eval(pl.struct(phone=pl.element(), word_index=pl.first().cum_count() - 1)))
        .explode("phones")
        .with_columns(pl.col("phones").struct.unnest())
        .explode("phone")
        .select("fileid", "phone", "word_index")
        .with_columns(pl.int_range(pl.len()).over("fileid").alias("index"))
    )
