import operator
from functools import reduce
from pathlib import Path

import polars as pl

from discophon.languages import all_languages


def lang_split(lang: str) -> pl.Expr:
    return (
        pl.when(pl.col(lang).is_null())
        .then(None)
        .otherwise(pl.col(lang).replace_strict({lang.iso_639_3: lang.split for lang in all_languages()}))
    )


def best_layer(df: pl.DataFrame, metric: str) -> pl.Expr:
    best_layers = (
        df.filter(pl.col("metric") == metric, pl.col("test_split") == "dev", pl.col("split") == "test")
        .group_by(["model", "layer", "duration"], maintain_order=True)
        .agg(pl.mean("score"), pl.len())
        .filter(pl.col("score") == pl.col("score").min().over(["model", "duration"]))
    )
    conditions = [
        (pl.col("model") == row["model"]) & (pl.col("duration") == row["duration"]) & (pl.col("layer") == row["layer"])
        for row in best_layers.iter_rows(named=True)
    ]
    return reduce(operator.or_, conditions, pl.lit(value=False))


def best_on_this_metric() -> pl.Expr:
    return pl.col("score") == pl.when(
        (pl.col("metric") == "per") | (pl.col("metric").str.starts_with("triphone_abx"))
    ).then(pl.min("score")).otherwise(pl.max("score"))


def native(train_col: str, test_col: str) -> pl.Expr:
    return pl.when(pl.col(train_col).is_null()).then(None).otherwise(pl.col(train_col) == pl.col(test_col))


def merge_metrics(df: pl.DataFrame, metrics: list[str]) -> pl.DataFrame:
    for metric in metrics:
        cols = [c for c in df.columns if c not in {"metric", "score", "std", "top"}]
        x = df.filter(pl.col("metric") == f"{metric}_within_speaker")
        y = df.filter(pl.col("metric") == f"{metric}_across_speaker")
        pair = (
            x.join(y, on=cols, suffix="_y", validate="1:1", nulls_equal=True)
            .with_columns(score=(pl.col("score") + pl.col("score_y")) / 2, metric=pl.lit(metric))
            .drop("score_y", "metric_y")
        )
        if len(pair) == 0:
            raise ValueError
        df = pl.concat([df, pair])
    return df


def read_scores(root: str | Path, *, best_layer_metric: str = "triphone_abx_continuous") -> pl.DataFrame:
    df = (
        pl.concat([pl.read_ndjson(p).with_columns(name=pl.lit(p.stem)) for p in Path(root).glob("*.jsonl")])
        .with_columns(pl.col("name").str.split("-").list.get(-1).cast(pl.Int64).alias("layer"))
        .with_columns(pl.col("name").str.strip_suffix("-" + pl.col("layer").cast(pl.Utf8)).alias("name"))
        .with_columns(
            pl.when(pl.col("name").str.contains("-ft-"))
            .then(pl.col("name").str.split_exact("-ft-", 1).struct.rename_fields(["model", "ft"]).struct.unnest())
            .otherwise(
                pl.struct([pl.col("name").alias("model"), pl.lit(None).cast(pl.Utf8).alias("ft")]).struct.unnest()
            )
        )
        .with_columns(
            pl.col("ft").str.split_exact("-train-", 1).struct.rename_fields(["ft_lang", "duration"]).struct.unnest()
        )
        .with_columns(pl.col("duration").fill_null(0))
        .select("split", "model", "ft_lang", "duration", "layer", "language", "metric", "score")
        .sort("split", "model", "ft_lang", "duration", "layer", "language", "metric", nulls_last=True)
        .with_columns(
            score=pl.col("score") * 100,
            ft_split=lang_split("ft_lang"),
            test_split=lang_split("language"),
            native=native("ft_lang", "language"),
            duration_val=pl.col("duration").replace_strict({"0": 1, "10min": 10, "1h": 60, "10h": 600}),
        )
        .filter(pl.col("native").is_null().or_(pl.col("native")))
    )
    df = merge_metrics(df, ["triphone_abx_discrete", "triphone_abx_continuous"])
    return df.with_columns(best_layer=best_layer(df, best_layer_metric))
