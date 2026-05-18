import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import altair as alt
import polars as pl
import polars.selectors as cs

from discophon.data import read_gold_annotations_as_dataframe
from discophon.languages import all_languages
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
MANIFEST_SPLITS = ["train-10min", "train-1h", "train-10h", "dev", "test"]

LANG_NAME_EXPR = (
    " : ".join(f"lang_sel.language == '{lang.iso_639_3}' ? '{lang.name}'" for lang in all_languages())
    + " : lang_sel.language"
)
METRIC_NAME_EXPR = (
    " : ".join(f"metric_sel.metric == '{k}' ? '{v}'" for k, v in METRICS.items()) + " : metric_sel.metric"
)
FT_NAME_EXPR = "ft_sel.duration == '0' ? 'pretrained' : ft_sel.duration + ' finetuning'"


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


def _language_select(value: str = "deu", *, field: str = "language") -> alt.Selection:
    return alt.selection_point(
        name="lang_sel",
        fields=[field],
        bind=alt.binding_radio(options=LANGUAGES, name="Language: "),
        value=value,
    )


def _split_select(options: list[str], value: str, *, name: str = "Split: ") -> alt.Selection:
    return alt.selection_point(
        name="split_sel",
        fields=["split"],
        bind=alt.binding_radio(options=options, name=name),
        value=value,
    )


def _path_meta_cols(path: Path) -> dict[str, pl.Expr]:
    parts = path.stem.split("-")
    return {"language": pl.lit(parts[1]), "split": pl.lit("-".join(parts[2:]))}


def _read_manifests(root: Path) -> pl.DataFrame:
    return pl.concat(
        [
            pl.read_csv(path, schema_overrides={"speaker": pl.String}).with_columns(**_path_meta_cols(path))
            for path in root.glob("*.csv")
        ]
    ).with_columns(duration=(pl.col("num_samples") / 16_000).round(3))


def _baseline_layers(
    base: alt.Chart,
    filters: Iterable[alt.Selection],
    legend_select: alt.Selection,
) -> tuple[alt.Chart, alt.Chart, alt.Chart]:
    filters = list(filters)
    fg = base.mark_line(point={"size": 50}).encode(
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
    for f in filters:
        fg = fg.transform_filter(f)
    fg = fg.add_params(legend_select)

    nearest = alt.selection_point(nearest=True, on="pointerover", fields=["layer"], empty=False)
    when_near = alt.when(nearest)
    rules = base
    for f in filters:
        rules = rules.transform_filter(f)
    rules = (
        rules.transform_filter(legend_select)
        .transform_pivot("model", value="score", groupby=["layer"])
        .mark_rule(color="gray", tooltip={"content": "data"})
        .encode(x="layer:Q", opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)))
        .add_params(nearest)
    )
    points = (
        fg.mark_point(size=100, filled=True)
        .encode(opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)))
        .transform_filter(legend_select)
    )
    return fg, rules, points


def plot_baselines_by_lang(data_url: str) -> alt.LayerChart | alt.FacetChart:
    metric_select, split_select, finetuning_select, legend_select = common_selectors()
    language_select = _language_select(LANGUAGES[0])
    share_y = alt.param(name="share_y", bind=alt.binding_checkbox(name="y-axis shared: "), value=False)

    base = alt.Chart(alt.UrlData(url=data_url, format=alt.CsvDataFormat()))
    bg_encoding = base.mark_point(opacity=0, size=0).encode(y=alt.Y("score:Q", scale=alt.Scale(zero=False)))
    bg_shared = bg_encoding.transform_filter(metric_select).transform_filter("share_y")
    bg_local = (
        bg_encoding.transform_filter(metric_select).transform_filter(language_select).transform_filter("!share_y")
    )
    fg, rules, points = _baseline_layers(
        base, [metric_select, finetuning_select, language_select, split_select], legend_select
    )
    title_expr = (
        f"({METRIC_NAME_EXPR}) + ' — ' + ({LANG_NAME_EXPR}) + ' — ' + split_sel.split + ' split — ' + ({FT_NAME_EXPR})"
    )
    return (
        alt.layer(bg_shared, bg_local, fg, points, rules)
        .add_params(metric_select, share_y, language_select, finetuning_select, split_select)
        .properties(width="container", height=300, title=alt.Title(text={"expr": title_expr}))
    )


def plot_baselines_by_split(data_url: str) -> alt.LayerChart | alt.FacetChart:
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
    fg, rules, points = _baseline_layers(
        base, [language_select, split_select, finetuning_select, metric_select], legend_select
    )
    title_expr = (
        f"({METRIC_NAME_EXPR}) + ' — ' + lang_sel.test_split + ' languages — ' "
        f"+ split_sel.split + ' split — ' + ({FT_NAME_EXPR})"
    )
    return (
        alt.layer(bg, fg, points, rules)
        .add_params(language_select, split_select, finetuning_select, metric_select)
        .properties(width="container", height=300, title=alt.Title(text={"expr": title_expr}))
    )


def plot_datasets_stats(root_manifests: Path) -> alt.Chart:
    bin_min, bin_max = 0, 35
    df = _read_manifests(root_manifests).select("language", "split", "duration")
    totals = df.group_by("language", "split", maintain_order=True).agg(total=pl.len())
    binned = (
        df.filter((pl.col("duration") >= bin_min) & (pl.col("duration") < bin_max))
        .with_columns(bin_start=pl.col("duration").floor().cast(pl.Int64))
        .group_by("language", "split", "bin_start", maintain_order=True)
        .agg(count=pl.len())
        .join(totals, on=["language", "split"])
        .with_columns(pct=pl.col("count") / pl.col("total"), bin_end=pl.col("bin_start") + 1)
        .with_columns(duration_range=pl.col("bin_start").cast(pl.String) + "-" + pl.col("bin_end").cast(pl.String))
        .select("language", "split", "bin_start", "bin_end", "duration_range", "pct")
    )
    language_select = _language_select()
    split_select = _split_select(MANIFEST_SPLITS, "test")

    base = alt.Chart(binned).transform_filter(language_select).transform_filter(split_select)
    x_enc = alt.X(
        "bin_start:Q",
        bin="binned",
        title="Duration (s)",
        scale=alt.Scale(domain=[bin_min, bin_max]),
        axis=alt.Axis(values=list(range(bin_max + 1)), grid=False),
    )
    bars = base.mark_bar().encode(
        x=x_enc,
        x2="bin_end:Q",
        y=alt.Y("pct:Q", title="Percentage (%)", axis=alt.Axis(format="%")),
    )
    overlay = base.mark_bar(opacity=0, binSpacing=0).encode(
        x=x_enc,
        x2="bin_end:Q",
        y=alt.Y("pct:Q"),
        tooltip=[
            alt.Tooltip("duration_range:N", title="Duration (s)"),
            alt.Tooltip("pct:Q", format=".2%", title="Percentage"),
        ],
    )
    title_expr = f"({LANG_NAME_EXPR}) + ' — ' + split_sel.split + ' split'"
    return (
        (bars + overlay)
        .add_params(split_select, language_select)
        .properties(width="container", height=300, title=alt.Title(text={"expr": title_expr}))
    )


def plot_speakers_stats(root_manifests: Path, *, top_speakers: int = 30) -> alt.Chart:
    per_speaker = (
        _read_manifests(root_manifests)
        .join(pl.read_ndjson(root_manifests / "manifest/speakers.jsonl").drop("split"), on=["speaker", "language"])
        .group_by(["language", "split", "speaker", "gender"], maintain_order=True)
        .agg(duration=pl.sum("duration"))
        .with_columns(speaker=pl.col("speaker").str.head(6))
    )
    totals = per_speaker.group_by(["language", "split"], maintain_order=True).agg(total=pl.len())
    gender_counts = (
        per_speaker.group_by(["language", "split", "gender"], maintain_order=True)
        .agg(total_gender=pl.len())
        .pivot(on="gender", index=["language", "split"], values="total_gender")
        .rename({"M": "count_M", "F": "count_F", "<unk>": "count_unk"})
        .fill_null(0)
    )
    df = (
        per_speaker.join(totals, on=["language", "split"])
        .join(gender_counts, on=["language", "split"])
        .sort("duration", descending=True)
        .with_columns(id=pl.struct("speaker", "duration", "gender", "total", "count_M", "count_F", "count_unk"))
        .group_by(["language", "split"], maintain_order=True)
        .agg(pl.head("id", n=top_speakers))
        .explode("id")
        .unnest("id")
        .with_columns(
            rank=pl.col("duration").rank("ordinal", descending=True).over("language", "split").cast(pl.Int32)
        )
    )
    language_select = _language_select()
    split_select = _split_select(MANIFEST_SPLITS, "test")
    title_expr = (
        f"({LANG_NAME_EXPR}) + ' — ' + split_sel.split + ' split — ' + data('data_0')[0].total + ' speakers (' "
        "+ data('data_0')[0].count_M + ' M, '"
        "+ data('data_0')[0].count_F + ' F, '"
        "+ data('data_0')[0].count_unk + ' <unk>)'"
    )
    return (
        alt.Chart(df)
        .transform_filter(language_select)
        .transform_filter(split_select)
        .mark_bar()
        .encode(
            x=alt.X("speaker:N", title="Speaker ID").sort("-y"),
            y=alt.Y("duration:Q", title="Total Duration (s)"),
            color=alt.Color(
                "gender:N",
                title="Gender",
                scale=alt.Scale(domain=["F", "M", "<unk>"], range=["#66C2A5FF", "#FC8D62FF", "#d3d3d3"]),
            ),
            tooltip=[
                alt.Tooltip("speaker:N", title="Speaker ID"),
                alt.Tooltip("duration:Q", title="Total Duration (s)", format=".1f"),
                alt.Tooltip("gender:N", title="Gender"),
            ],
        )
        .add_params(split_select, language_select)
        .properties(width="container", height=300, title=alt.Title(text={"expr": title_expr}))
    )


def plot_phone_distribution(root: Path) -> alt.Chart:
    df = (
        pl.concat(
            [
                read_gold_annotations_as_dataframe(path).with_columns(**_path_meta_cols(path))
                for path in root.glob("*.txt")
            ]
        )
        .group_by(["language", "split", "#phone"])
        .agg(count=pl.len())
        .sort("language", "split", "#phone")
    )
    df = df.join(
        df.filter(pl.col("split") == "test")
        .rename({"count": "test_count"})
        .select("language", "#phone", "test_count"),
        on=["language", "#phone"],
        how="left",
    ).with_columns(pl.col("test_count").fill_null(0))
    language_select = _language_select()
    split_select = _split_select(["dev", "test"], "test")

    phone_counts = {lang.iso_639_3: {} for lang in all_languages()}
    for row in (
        df.group_by(["language", "split"], maintain_order=True)
        .agg(pl.col("#phone").n_unique().alias("n"))
        .iter_rows(named=True)
    ):
        phone_counts[row["language"]][row["split"]] = row["n"] - 1
    title_expr = (
        f"({LANG_NAME_EXPR}) + ' — ' + split_sel.split + ' split'"
        f" + ' (' + {json.dumps(phone_counts)}[lang_sel.language][split_sel.split] + ' phonemes)'"
    )
    bars = (
        alt.Chart(df)
        .transform_filter(language_select)
        .transform_filter(split_select)
        .mark_bar()
        .encode(
            x=alt.X("#phone:N", title="Phone", axis=alt.Axis(labelAngle=0)).sort(
                field="test_count", op="max", order="descending"
            ),
            y=alt.Y("count:Q", title="Count"),
            tooltip=[alt.Tooltip("#phone:N", title="Phone"), alt.Tooltip("count:Q", title="Count")],
        )
    )
    domain_anchor = (
        alt.Chart(df.group_by("language").agg(pl.col("count").max().alias("max_count")))
        .transform_filter(language_select)
        .mark_point(opacity=0, size=0)
        .encode(y=alt.Y("max_count:Q"))
    )
    return (
        alt.layer(bars, domain_anchor)
        .add_params(split_select, language_select)
        .resolve_scale(y="shared")
        .properties(width="container", height=300, title=alt.Title(text={"expr": title_expr}))
    )


def to_html(chart: alt.Chart | alt.LayerChart | alt.FacetChart, css: str) -> str:
    return chart.to_html(embed_options={"actions": False}).replace(
        "</style>", f'</style>\n  <link rel="stylesheet" href="{css}">', 1
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Path to the benchmark dataset")
    parser.add_argument("scores", type=Path, help="Path to the baseline scores")
    parser.add_argument("destination", type=Path, help="Path to the assets directory in docs")
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

    def write(chart: alt.Chart | alt.LayerChart | alt.FacetChart, name: str) -> None:
        (args.destination / name).write_text(to_html(chart, "../stylesheets/vega.css"), encoding="utf-8")

    write(plot_baselines_by_lang("scores_by_lang.csv"), "baseline_across_layers_by_lang.html")
    write(plot_baselines_by_split("scores_by_split.csv"), "baseline_across_layers_by_split.html")
    write(plot_datasets_stats(args.dataset / "manifest"), "dataset_stats.html")
    write(plot_speakers_stats(args.dataset / "manifest"), "speaker_stats.html")
    write(plot_phone_distribution(args.dataset / "alignment"), "phone_distribution.html")
