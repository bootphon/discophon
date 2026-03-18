# %%
from pathlib import Path

import polars as pl
import polars.selectors as cs
from discophon.languages import dev_languages

MODELS = {
    "spidr-mmsulab": "SpidR MMS-ulab",
    "spidr-vp20": "SpidR VP-20",
    "hubert-mmsulab-it2": "HuBERT MMS-ulab",
    "hubert-vp20-it2": "HuBERT VP-20",
}
METRICS = ["per", "r_val"]


def merge_metrics(df: pl.DataFrame, metric: str) -> pl.DataFrame:
    m1, m2 = f"{metric}_within_speaker", f"{metric}_across_speaker"
    cols = [c for c in df.columns if c not in {"metric", "score", "std"}]
    x = df.filter(pl.col("metric") == m1)
    y = df.filter(pl.col("metric") == m2)
    z = df.filter(~pl.col("metric").is_in([m1, m2]))
    pair = (
        x.join(y, on=cols, suffix="_y", validate="1:1")
        .with_columns(
            score=(pl.col("score") + pl.col("score_y")) / 2, metric=pl.lit(metric)
        )
        .drop("score_y", "metric_y")
    )
    return pl.concat([pair, z])


def read_data(p: Path) -> pl.DataFrame:
    df = pl.read_ndjson(p)
    if "-ft-" in p.stem:
        model, ft_duration_layer = p.stem.split("-ft-")
        ft, *split, layer = ft_duration_layer.split("-")
        assert split == ["train", "10h"]
        duration = "10h"
    else:
        *model, layer = p.stem.split("-")
        model = "-".join(model)
        ft, duration = None, "0"
    return df.with_columns(
        pl.lit(model, dtype=pl.String).alias("model"),
        pl.lit(ft, dtype=pl.String).alias("ft"),
        pl.lit(duration, dtype=pl.String).alias("duration"),
        pl.lit(layer, dtype=pl.Int64).alias("layer"),
    )


def set_languages() -> pl.Expr:
    return (
        pl.when(pl.col("language").is_in([lang.iso_639_3 for lang in dev_languages()]))
        .then(pl.lit("dev"))
        .otherwise(pl.lit("test"))
        .alias("set")
    )


def format_row(entry: dict[str, float | str]) -> str:
    row = rf"{MODELS[entry['model']]} & "  # (L{entry['layer']}) & "
    for lang_set in ["dev", "test"]:
        for metric in METRICS:
            score = entry["score_{" + f'"{lang_set}","{metric}"' + "}"]
            # std = entry[f"std_\{{set}_{metric}_std"]
            if entry["top_{" + f'"{lang_set}","{metric}"' + "}"]:
                row += rf"$\mathbf{{{score:.2f}}}$ & "
            else:
                row += rf"{score:.2f} & "
    return row[:-2] + r"\\"


def print_tabular(df: pl.DataFrame) -> str:
    lines = []
    lines.append(r"\begin{tabular}{lcccc}")
    lines.append(r"\toprule")
    lines.append(
        r"& \multicolumn{2}{c}{\dev languages} & \multicolumn{2}{c}{\test languages} \\"
    )
    lines.append(r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}")
    lines.append(r"& \bf PER $\downarrow$ & \bf$\bm R$-val. $\uparrow$ " * 2 + r"\\")
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


df = (
    (
        merge_metrics(
            pl.concat(
                [read_data(p) for p in Path("./data/scores_fixed/").rglob("*.jsonl")]
            )
            .with_columns(set_languages())
            .filter(pl.col("model").is_in(MODELS), pl.col("split") == "test")
            .with_columns(pl.col("score") * 100)
            .group_by(cs.exclude("score", "language", "ft"), maintain_order=True)
            .agg(score=pl.mean("score"))
            .filter(pl.col("duration").is_in(["0", "10h"])),
            "triphone_abx_discrete",
        ).filter(pl.col("metric").is_in(METRICS))
    )
    .with_columns(
        top=(
            pl.col("score")
            == pl.when(pl.col("metric").is_in(["per", "triphone_abx_discrete"]))
            .then(pl.min("score"))
            .otherwise(pl.max("score"))
        ).over(["duration", "metric", "set"])
    )
    .pivot(on=["set", "metric"], index=["split", "model", "duration", "layer"])
    .sort("model")
)

print(print_tabular(df))
# %%
df
# %%
df
# %%
