# %%
import polars as pl
import polars.selectors as cs
from discophon.languages import dev_languages

MODELS = {
    "spidr-mmsulab": "SpidR MMS-ulab",
    "spidr-vp20": "SpidR VP-20",
    "hubert-mmsulab-it2": "HuBERT MMS-ulab",
    "hubert-vp20-it2": "HuBERT VP-20",
}
METRICS = {
    "per": r"\bf PER $\downarrow$",
    "r_val": r"\bf$\bm R$-value $\uparrow$",
    "f1": r"$F_1$ $\uparrow$",
    "pnmi": r"PNMI $\uparrow$",
    "triphone_abx_continuous": r"ABX c. $\downarrow$",
    # "triphone_abx_discrete": r"ABX d. $\downarrow$",
}
TARGET_METRIC = "triphone_abx_discrete_within_speaker"


def set_languages() -> pl.Expr:
    return (
        pl.when(pl.col("language").is_in([lang.iso_639_3 for lang in dev_languages()]))
        .then(pl.lit("dev"))
        .otherwise(pl.lit("test"))
        .alias("set")
    )


def merge_metrics(df: pl.DataFrame, metrics: list[str]) -> pl.DataFrame:
    for metric in metrics:
        cols = [c for c in df.columns if c not in {"metric", "score", "std", "top"}]
        x = df.filter(pl.col("metric") == f"{metric}_within_speaker")
        y = df.filter(pl.col("metric") == f"{metric}_across_speaker")
        pair = (
            x.join(y, on=cols, suffix="_y", validate="1:1", nulls_equal=True)
            .with_columns(
                score=(pl.col("score") + pl.col("score_y")) / 2, metric=pl.lit(metric)
            )
            .drop("score_y", "metric_y")
        )
        assert len(pair) > 0
        df = pl.concat([df, pair])
    return df


def format_row(entry: dict[str, float | str]) -> str:
    row = rf"{MODELS[entry['model']]} (L{entry['layer']}) & "
    for lang_set in ["dev", "test"]:
        for metric in METRICS:
            score = entry["score_{" + f'"{lang_set}","{metric}"' + "}"]
            # std = entry["std_{" + f'"{lang_set}","{metric}"' + "}"]
            if entry["top_{" + f'"{lang_set}","{metric}"' + "}"]:
                row += rf"$\mathbf{{{score:.2f}}}$ & "
            else:
                row += rf"{score:.2f} & "
    return row[:-2] + r"\\"


def print_tabular(df: pl.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{tabular}{lcccccccccc}")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{5}{c}{\dev languages} & \multicolumn{5}{c}{\test languages} \\"
    )
    lines.append(r"\cmidrule(lr){2-6} \cmidrule(lr){7-11}")
    lines.append(" ".join([rf"& {METRICS[metric]} " for metric in METRICS] * 2) + r"\\")
    lines.append(r"\midrule")
    lines.append(r"\textbf{Zero-shot} \\")
    for row in df.filter(pl.col("duration") == "0").iter_rows(named=True):
        lines.append(format_row(row))
    lines.append(r"\addlinespace")
    lines.append(r"\textbf{Finetuned on 10h} \\")
    for row in df.filter(pl.col("duration") == "10h").iter_rows(named=True):
        lines.append(format_row(row))
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    return "\n".join(lines)


if __name__ == "__main__":
    path = "./data/final.jsonl"
    full = merge_metrics(
        pl.read_ndjson(path)
        .with_columns(set_languages())
        .filter(pl.col("model").is_in(MODELS), pl.col("split") == "test"),
        ["triphone_abx_continuous", "triphone_abx_discrete"],
    )
    best_layers = (
        full.group_by(cs.exclude("score", "ft", "language"), maintain_order=True)
        .agg(pl.mean("score"))
        .filter(pl.col("set") == "dev", pl.col("metric") == TARGET_METRIC)
        .filter(
            pl.col("score") == pl.col("score").min().over(["model", "duration"]),
        )
        .drop("split", "set", "score", "metric")
    )
    df = (
        full.join(best_layers, on=["model", "duration", "layer"])
        .with_columns(pl.col("score") * 100)
        .group_by(cs.exclude("score", "language", "ft"), maintain_order=True)
        .agg(score=pl.mean("score"), std=pl.std("score"))
        .filter(pl.col("duration").is_in(["0", "10h"]))
        .with_columns(
            top=(
                pl.col("score")
                == pl.when(
                    (pl.col("metric") == "per")
                    | (pl.col("metric").str.starts_with("triphone_abx"))
                )
                .then(pl.min("score"))
                .otherwise(pl.max("score"))
            ).over(["duration", "metric", "set"])
        )
        .filter(pl.col("metric").is_in(METRICS))
        .pivot(on=["set", "metric"], index=["split", "model", "duration", "layer"])
    )
    with open("./tables/benchmark-many-to-one.tex", "w") as f:
        f.write(print_tabular(df))

# %%
# %%
df.filter(pl.col("metric") == "triphone_abx_continuous_across_speaker")
# %%
df["metric"].unique().to_list()
# %%
df["model"].value_counts()
# %%
full["duration"].value_counts()
# %%
full
# %%
metric = "triphone_abx_continuous"
cols = [c for c in full.columns if c not in {"metric", "score", "std", "top"}]
x = full.filter(pl.col("metric") == f"{metric}_within_speaker")
y = full.filter(pl.col("metric") == f"{metric}_across_speaker")
pair = (
    x.join(y, on=cols, suffix="_y", validate="1:1")
    .with_columns(
        score=(pl.col("score") + pl.col("score_y")) / 2, metric=pl.lit(metric)
    )
    .drop("score_y", "metric_y")
)
assert len(pair) > 0

# %%
y["duration"].value_counts()
# %%
a = x.filter(pl.col("duration") == "0").head()
b = y.filter(pl.col("duration") == "0").head()

# %%
a.join(b, on=cols, suffix="_y", validate="1:1", nulls_equal=True)[
    "duration"
].value_counts()
# %%
x.filter(pl.col("duration") == "0")
# %%
y.filter(pl.col("duration") == "0")
# %%
cols
# %%
a
# %%
b
# %%
a
# %%
b
# %%
