import argparse
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from discophon.data import read_scores
from discophon.leaderboard import read_models

METRICS = {
    "per": r"\bf PER $\downarrow$",
    "r_val": r"\bf$\bm R$-value $\uparrow$",
    "f1": r"$F_1$ $\uparrow$",
    "pnmi": r"PNMI $\uparrow$",
    "triphone_abx_continuous": r"ABX c. $\downarrow$",
}


def best_on_this_metric() -> pl.Expr:
    lower_is_better = (pl.col("metric") == "per") | pl.col("metric").str.starts_with("triphone_abx")
    return pl.col("score") == pl.when(lower_is_better).then(pl.min("score")).otherwise(pl.max("score"))


def average(scores: pl.DataFrame, folders: list[str], metrics: Iterable[str]) -> pl.DataFrame:
    """Scores at the best layer, averaged over dev and test languages, one row per (model, duration)."""
    return (
        scores.filter(pl.col("folder").is_in(folders), pl.col("metric").is_in(list(metrics)))
        .group_by("model", "layer", "duration", "language_split", "metric", maintain_order=True)
        .agg(pl.mean("score"))
        .with_columns(top=best_on_this_metric().over("metric", "duration", "language_split"))
        .pivot(on=["language_split", "metric"], index=["model", "layer", "duration"])
    )


def format_row(entry: dict, metrics: Iterable[str], *, with_layer: bool) -> str:
    row = rf"{entry['model']} (L{entry['layer']}) & " if with_layer else rf"{entry['model']} & "
    for lang_set in ["dev", "test"]:
        for metric in metrics:
            score = entry["score_{" + f'"{lang_set}","{metric}"' + "}"]
            if entry["top_{" + f'"{lang_set}","{metric}"' + "}"]:
                row += rf"$\mathbf{{{score:.2f}}}$ & "
            else:
                row += rf"{score:.2f} & " if score is not None else "N/A & "
    return row[:-2] + r"\\"


def get_tabular(df: pl.DataFrame, metrics: dict[str, str], *, with_layer: bool = True) -> str:
    ncols = len(metrics)
    lines = [
        r"\begin{tabular}{l" + 2 * ncols * "c" + "}",
        r"\toprule",
        r"& \multicolumn{"
        + str(ncols)
        + r"}{c}{\dev languages} & \multicolumn{"
        + str(ncols)
        + r"}{c}{\test languages} \\",
        r"\cmidrule(lr){2-" + str(ncols + 1) + r"} \cmidrule(lr){" + f"{ncols + 2}-{2 * ncols + 1}" + r"}",
        " ".join([rf"& {name} " for name in metrics.values()] * 2) + r"\\",
        r"\midrule",
        r"\textbf{Zero-shot} \\",
    ]
    lines += [
        format_row(row, metrics, with_layer=with_layer)
        for row in df.filter(pl.col("duration") == "0").iter_rows(named=True)
    ]
    lines += [r"\addlinespace", r"\textbf{Finetuned on 10h} \\"]
    lines += [
        format_row(row, metrics, with_layer=with_layer)
        for row in df.filter(pl.col("duration") == "10h").iter_rows(named=True)
    ]
    lines += [r"\bottomrule", r"\end{tabular}"]
    return "\n".join(lines)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("artifacts", type=Path, help="Path to the artifacts dataset")
    parser.add_argument("--leaderboard", type=Path, default=Path("../leaderboard"), help="Path to the leaderboard")
    args = parser.parse_args()
    baselines = {
        key: entry["label"] for key, entry in read_models(args.leaderboard).items() if entry["category"] == "baseline"
    }

    scores = (
        read_scores(args.artifacts)
        .filter(
            pl.col("split") == "test",
            pl.col("ft_lang").is_null() | (pl.col("ft_lang") == pl.col("language")),
            pl.col("duration").is_in(["0", "10h"]),
            pl.col("model").is_in(baselines),
        )
        .with_columns(pl.col("model").replace_strict(baselines), pl.col("score") * 100)
        .sort("model", "duration", "layer", "language")
    )
    best = (  # Layer with the lowest continuous ABX on dev languages, for each model and duration
        scores.filter(pl.col("metric") == "triphone_abx_continuous", pl.col("language_split") == "dev")
        .group_by("model", "duration", "layer")
        .agg(pl.mean("score"))
        .sort("score", "layer")
        .group_by("model", "duration")
        .first()
        .select("model", "duration", "layer")
    )
    scores = scores.join(best, on=["model", "duration", "layer"], maintain_order="left")
    print(get_tabular(average(scores, ["many_to_one", "continuous"], METRICS), METRICS))
    one_to_one_metrics = {"per": r"\bf PER $\downarrow$", "r_val": r"\bf$\bm R$-value $\uparrow$"}
    print(get_tabular(average(scores, ["one_to_one"], one_to_one_metrics), one_to_one_metrics, with_layer=False))
