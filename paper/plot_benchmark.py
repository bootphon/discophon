import argparse
from pathlib import Path

import altair as alt
import polars as pl
import polars.selectors as cs

from discophon.languages import all_languages, get_language
from discophon.paper import merge_metrics, read_scores

MODELS = {
    "spidr-mmsulab": "SpidR MMS-ulab",
    "spidr-vp20": "SpidR VP-20",
    "hubert-mmsulab-it2": "HuBERT MMS-ulab",
    "hubert-vp20-it2": "HuBERT VP-20",
}
LANGUAGES = [lang.iso_639_3 for lang in all_languages()]
METRICS = {
    "per": "PER",
    "pnmi": "PNMI",
    "f1": "F1",
    "r_val": "R-val",
    "triphone_abx_continuous": "ABX c.",
    "triphone_abx_discrete": "ABX d.",
}


def common_selectors() -> tuple[alt.Selection, alt.Selection, alt.Selection, alt.Selection]:
    metric_select = alt.selection_point(
        name="metric_sel",
        fields=["metric"],
        bind=alt.binding_radio(options=list(METRICS.keys()), labels=list(METRICS.values()), name="Metric: "),
        value=next(iter(METRICS.keys())),
    )
    split_select = alt.selection_point(
        name="split_sel",
        fields=["split"],
        bind=alt.binding_radio(options=["dev", "test"], name="Split: "),
        value="test",
    )
    finetuning_select = alt.selection_point(
        name="ft_sel",
        fields=["duration"],
        bind=alt.binding_radio(options=["0", "10min", "1h", "10h"], name="Finetuning: "),
        value="10h",
    )
    legend_select = alt.selection_point(fields=["model"], bind="legend", toggle="true")
    return metric_select, split_select, finetuning_select, legend_select


def plot_by_lang(data_url: str) -> alt.LayerChart:  # noqa: PLR0914
    metric_select, split_select, finetuning_select, legend_select = common_selectors()
    language_select = alt.selection_point(
        name="lang_sel",
        fields=["language"],
        bind=alt.binding_radio(options=LANGUAGES, name="Language: "),
        value=LANGUAGES[0],
    )
    share_y = alt.param(name="share_y", bind=alt.binding_checkbox(name="y-axis shared: "), value=False)

    base = alt.Chart(alt.UrlData(url=data_url, format=alt.CsvDataFormat()))
    bg_shared = (
        base.mark_point(opacity=0, size=0)
        .encode(y=alt.Y("score:Q", scale=alt.Scale(zero=False)))
        .transform_filter(metric_select)
        .transform_filter("share_y")
    )
    bg_local = (
        base.mark_point(opacity=0, size=0)
        .encode(y=alt.Y("score:Q", scale=alt.Scale(zero=False)))
        .transform_filter(metric_select)
        .transform_filter(language_select)
        .transform_filter("!share_y")
    )
    fg = (
        base.mark_line(point={"size": 50})
        .encode(
            x=alt.X("layer:Q", title="Layer", axis=alt.Axis(grid=False, format="d")),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "model:N",
                title="Model",
                sort=list(MODELS.values()),
                legend=alt.Legend(columns=2, direction="horizontal", titleAnchor="middle", orient="top"),
            ),
            opacity=alt.condition(legend_select, alt.value(1), alt.value(0.1)),
            strokeWidth=alt.condition(legend_select, alt.value(2.5), alt.value(2)),
        )
        .transform_filter(metric_select)
        .transform_filter(finetuning_select)
        .transform_filter(language_select)
        .transform_filter(split_select)
        .add_params(legend_select)
    )
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["layer"], empty=False)
    when_near = alt.when(nearest)
    rules = (
        base.transform_filter(metric_select)
        .transform_filter(finetuning_select)
        .transform_filter(language_select)
        .transform_filter(split_select)
        .transform_filter(legend_select)
        .transform_pivot("model", value="score", groupby=["layer"])
        .mark_rule(color="gray", tooltip={"content": "data"})
        .encode(
            x="layer:Q",
            opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
        )
        .add_params(nearest)
    )
    points = (
        fg.mark_point(size=100, filled=True)
        .encode(opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)))
        .transform_filter(legend_select)
    )
    metric_ex = " : ".join(f"metric_sel.metric == '{k}' ? '{v}'" for k, v in METRICS.items()) + " : metric_sel.metric"
    lang_ex = (
        " : ".join(f"lang_sel.language == '{iso}' ? '{get_language(iso).name}'" for iso in LANGUAGES)
        + " : lang_sel.language"
    )
    ft_ex = "ft_sel.duration == '0' ? 'pretrained' : ft_sel.duration + ' finetuning'"
    title_ex = f"({metric_ex}) + ' — ' + ({lang_ex}) + ' — ' + split_sel.split + ' split — ' + ({ft_ex})"
    return (
        alt.layer(bg_shared, bg_local, fg, points, rules)
        .add_params(metric_select, share_y, language_select, finetuning_select, split_select)
        .properties(width="container", height=300, title=alt.Title(text={"expr": title_ex}))
    )


def plot_by_split(data_url: str) -> alt.LayerChart:
    metric_select, split_select, finetuning_select, legend_select = common_selectors()
    language_select = alt.selection_point(
        name="lang_sel",
        fields=["test_split"],
        bind=alt.binding_radio(options=["dev", "test"], name="Languages: "),
        value="test",
    )
    base = alt.Chart(alt.UrlData(url=data_url, format=alt.CsvDataFormat()))
    bg = (
        base.mark_point(opacity=0, size=0)
        .encode(y=alt.Y("score:Q", scale=alt.Scale(zero=False)))
        .transform_filter(metric_select)
    )
    fg = (
        base.mark_line(point={"size": 50})
        .encode(
            x=alt.X("layer:Q", title="Layer", axis=alt.Axis(grid=False, format="d")),
            y=alt.Y("score:Q", title="Score", scale=alt.Scale(zero=False)),
            color=alt.Color(
                "model:N",
                title="Model",
                sort=list(MODELS.values()),
                legend=alt.Legend(columns=2, direction="horizontal", titleAnchor="middle", orient="top"),
            ),
            opacity=alt.condition(legend_select, alt.value(1), alt.value(0.1)),
            strokeWidth=alt.condition(legend_select, alt.value(2.5), alt.value(2)),
        )
        .transform_filter(language_select)
        .transform_filter(split_select)
        .transform_filter(finetuning_select)
        .transform_filter(metric_select)
        .add_params(legend_select)
    )
    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["layer"], empty=False)
    when_near = alt.when(nearest)
    rules = (
        base.transform_filter(language_select)
        .transform_filter(split_select)
        .transform_filter(finetuning_select)
        .transform_filter(metric_select)
        .transform_filter(legend_select)
        .transform_pivot("model", value="score", groupby=["layer"])
        .mark_rule(color="gray", tooltip={"content": "data"})
        .encode(
            x="layer:Q",
            opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)),
        )
        .add_params(nearest)
    )
    points = (
        fg.mark_point(size=100, filled=True)
        .encode(opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)))
        .transform_filter(legend_select)
    )
    metric_ex = " : ".join(f"metric_sel.metric == '{k}' ? '{v}'" for k, v in METRICS.items()) + " : metric_sel.metric"
    ft_ex = "ft_sel.duration == '0' ? 'pretrained' : ft_sel.duration + ' finetuning'"
    title_expr = (
        f"({metric_ex}) + ' — ' + lang_sel.test_split + ' languages — ' + split_sel.split + ' split — ' + ({ft_ex})"
    )
    return (
        alt.layer(bg, fg, points, rules)
        .add_params(language_select, split_select, finetuning_select, metric_select)
        .properties(width="container", height=300, title=alt.Title(text={"expr": title_expr}))
    )


def add_css(html: str, css: str) -> str:
    return html.replace("</style>", f'</style>\n  <link rel="stylesheet" href="{css}">', 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    by_lang = (
        merge_metrics(read_scores(args.scores), ["triphone_abx_discrete", "triphone_abx_continuous"])
        .filter(
            pl.col("native").is_null().or_(pl.col("native")),
            pl.col("model").is_in(MODELS),
            pl.col("metric").is_in(METRICS),
        )
        .with_columns(pl.col("model").replace_strict(MODELS))
        .select("split", "model", "layer", "duration", "duration_val", "language", "test_split", "metric", "score")
    )
    by_split = by_lang.group_by(cs.exclude("language", "score"), maintain_order=True).agg(pl.mean("score"))
    by_lang.with_columns(pl.col("score").round(2)).write_csv(args.destination / "scores_by_lang.csv")
    by_split.with_columns(pl.col("score").round(2)).write_csv(args.destination / "scores_by_split.csv")

    html = plot_by_lang("scores_by_lang.csv").to_html(embed_options={"actions": False})
    html = add_css(html, "../stylesheets/vega.css")
    Path(args.destination / "baseline_across_layers_by_lang.html").write_text(html, encoding="utf-8")

    html = plot_by_split("scores_by_split.csv").to_html(embed_options={"actions": False})
    html = add_css(html, "../stylesheets/vega.css")
    Path(args.destination / "baseline_across_layers_by_split.html").write_text(html, encoding="utf-8")
