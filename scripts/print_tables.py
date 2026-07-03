import argparse
from collections.abc import Iterable
from pathlib import Path

import polars as pl

from discophon.paper import best_on_this_metric, read_scores

METRICS = {
    "per": r"\bf PER $\downarrow$",
    "r_val": r"\bf$\bm R$-value $\uparrow$",
    "f1": r"$F_1$ $\uparrow$",
    "pnmi": r"PNMI $\uparrow$",
    "triphone_abx_continuous": r"ABX c. $\downarrow$",
}

MODELS = {
    "spidr-mmsulab": "SpidR MMS-ulab",
    "spidr-vp20": "SpidR VP-20",
    "hubert-mmsulab-it2": "HuBERT MMS-ulab",
    "hubert-vp20-it2": "HuBERT VP-20",
}


def format_row(entry: dict, metrics: Iterable[str], *, with_layer: bool) -> str:
    row = rf"{MODELS[entry['model']]} (L{entry['layer']}) & " if with_layer else rf"{MODELS[entry['model']]} & "
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
    parser.add_argument("many_to_one", type=Path)
    parser.add_argument("one_to_one", type=Path)
    args = parser.parse_args()

    many_to_one = (
        read_scores(args.many_to_one)
        .filter(
            pl.col("split") == "test",
            pl.col("best_layer"),
            pl.col("metric").is_in(METRICS),
            pl.col("model").is_in(MODELS),
        )
        .select("model", "layer", "duration", "language", "test_split", "metric", "score")
        .group_by(["model", "layer", "duration", "test_split", "metric"], maintain_order=True)
        .agg(pl.mean("score"))
        .with_columns(top=best_on_this_metric().over(["metric", "duration", "test_split"]))
        .pivot(on=["test_split", "metric"], index=["model", "layer", "duration"])
    )
    print(get_tabular(many_to_one, METRICS))

    best_layers = {
        "hubert-mmsulab-it2": 12,
        "hubert-vp20-it2": 11,
        "spidr-mmsulab": 5,
        "spidr-vp20": 6,
    }

    one_to_one = (
        read_scores(args.one_to_one)
        .filter(
            pl.col("split") == "test",
            pl.col("metric").is_in(["per", "r_val"]),
            pl.col("model").is_in(MODELS),
        )
        .filter(pl.col("layer") == pl.col("model").replace_strict(best_layers))
        .select("model", "layer", "duration", "language", "test_split", "metric", "score")
        .group_by(["model", "layer", "duration", "test_split", "metric"], maintain_order=True)
        .agg(pl.mean("score"))
        .with_columns(top=best_on_this_metric().over(["metric", "duration", "test_split"]))
        .pivot(on=["test_split", "metric"], index=["model", "layer", "duration"])
    )
    print(
        get_tabular(
            one_to_one, {"per": r"\bf PER $\downarrow$", "r_val": r"\bf$\bm R$-value $\uparrow$"}, with_layer=False
        )
    )
