import argparse
import json
from collections.abc import Iterable
from pathlib import Path

import altair as alt
import polars as pl

from discophon.data import read_gold_annotations_as_dataframe, read_scores
from discophon.languages import all_languages

MODELS = {
    "spidr-mmsulab": "SpidR MMS-ulab",
    "spidr-vp20": "SpidR VP-20",
    "hubert-mmsulab-it2": "HuBERT MMS-ulab",
    "hubert-vp20-it2": "HuBERT VP-20",
}
LANGUAGES = [lang.iso_639_3 for lang in all_languages()]
LANGUAGE_NAMES = {"avg-test": "Average over test languages", "avg-dev": "Average over dev languages"} | {
    lang.iso_639_3: lang.name for lang in all_languages()
}
METRICS = {
    "per": "PER",
    "pnmi": "PNMI",
    "f1": "F1",
    "r_val": "R-val",
    "triphone_abx_continuous": "ABX c.",
    "triphone_abx_discrete": "ABX d.",
}
MANIFEST_SPLITS = ["train-10min", "train-1h", "train-10h", "dev", "test"]
DURATION_MINUTES = {"0": 1, "10min": 10, "1h": 60, "10h": 600}

LANG_NAME_EXPR = (
    " : ".join(f"lang_sel.language == '{lang.iso_639_3}' ? '{lang.name}'" for lang in all_languages())
    + " : lang_sel.language"
)
SCORES_LANG_NAME_EXPR = (
    " : ".join(f"lang_sel.language == '{iso}' ? '{name}'" for iso, name in LANGUAGE_NAMES.items())
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
        bind=alt.binding_radio(options=list(METRICS.keys()), labels=list(METRICS.values()), name="Metric"),
        value=next(iter(METRICS.keys())),
    )
    language_select = alt.selection_point(
        name="lang_sel",
        fields=["language"],
        bind=alt.binding_select(
            options=list(LANGUAGE_NAMES.keys()), labels=list(LANGUAGE_NAMES.values()), name="Language"
        ),
        value="avg-test",
    )
    finetuning_select = alt.selection_point(
        name="ft_sel",
        fields=["duration"],
        bind=alt.binding_radio(options=["0", "10min", "1h", "10h"], name="Finetuning"),
        value="10h",
    )
    legend_select = alt.selection_point(fields=["model"], bind="legend", toggle="true")
    return metric_select, language_select, finetuning_select, legend_select


def _language_select(value: str = "deu") -> alt.Selection:
    return alt.selection_point(
        name="lang_sel",
        fields=["language"],
        bind=alt.binding_radio(options=LANGUAGES, name="Language"),
        value=value,
    )


def _split_select(options: list[str], value: str, *, name: str = "Split") -> alt.Selection:
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


_LAYER_X = ("layer", alt.X("layer:Q", title="Layer", axis=alt.Axis(grid=False, format="d")))
_DURATION_X = (
    "duration_minutes",
    alt.X(
        "duration_minutes:Q",
        title="Finetuning duration",
        scale=alt.Scale(type="log", base=10),
        axis=alt.Axis(
            grid=False,
            values=[1, 10, 60, 600],
            labelExpr="datum.value == 1 ? '0' : datum.value == 10 ? '10min' : datum.value == 60 ? '1h' : '10h'",
        ),
    ),
)


def _baseline_layers(
    base: alt.Chart,
    filters: Iterable[alt.Selection],
    legend_select: alt.Selection,
    *,
    x: tuple[str, alt.X] = _LAYER_X,
) -> tuple[alt.Chart, alt.Chart, alt.Chart]:
    filters = list(filters)
    x_field, x_enc = x
    fg = base.mark_line(point={"size": 50}).encode(
        x=x_enc,
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

    nearest = alt.selection_point(nearest=True, on="pointerover", fields=[x_field], empty=False)
    when_near = alt.when(nearest)
    rules = base
    for f in filters:
        rules = rules.transform_filter(f)
    rules = (
        rules.transform_filter(legend_select)
        .transform_pivot("model", value="score", groupby=[x_field])
        .mark_rule(color="gray", tooltip={"content": "data"})
        .encode(x=f"{x_field}:Q", opacity=when_near.then(alt.value(0.3)).otherwise(alt.value(0)))
        .add_params(nearest)
    )
    points = (
        fg.mark_point(size=100, filled=True)
        .encode(opacity=when_near.then(alt.value(1)).otherwise(alt.value(0)))
        .transform_filter(legend_select)
    )
    return fg, rules, points


def baseline_scores(artifacts: Path) -> pl.DataFrame:
    """Scores (in %) of the baselines on the test split, for each language and averaged over dev and test languages.

    Finetuned models are evaluated on their finetuning language. The best layer of each model and duration
    minimizes the continuous ABX on dev languages.
    """
    df = (
        read_scores(artifacts)
        .filter(
            pl.col("split") == "test",
            pl.col("ft_lang").is_null() | (pl.col("ft_lang") == pl.col("language")),
            pl.col("model").is_in(MODELS),
            pl.col("folder").is_in(["many_to_one", "continuous"]),
            pl.col("metric").is_in(METRICS),
        )
        .with_columns(pl.col("score") * 100, duration_minutes=pl.col("duration").replace_strict(DURATION_MINUTES))
    )
    best = (
        df.filter(pl.col("metric") == "triphone_abx_continuous", pl.col("language_split") == "dev")
        .group_by("model", "duration", "layer")
        .agg(pl.mean("score"))
        .sort("score", "layer")
        .group_by("model", "duration")
        .first()
        .select("model", "duration", "layer", best_layer=pl.lit(value=True))
    )
    df = df.join(best, on=["model", "duration", "layer"], how="left").with_columns(
        pl.col("best_layer").fill_null(value=False)
    )
    columns = ["model", "layer", "duration", "duration_minutes", "language", "metric", "score", "best_layer"]
    averages = (
        df.group_by("model", "layer", "duration", "duration_minutes", "metric", "best_layer", "language_split")
        .agg(pl.mean("score"))
        .with_columns(language="avg-" + pl.col("language_split"))
    )
    return (
        pl.concat([df.select(columns), averages.select(columns)])
        .with_columns(pl.col("model").replace_strict(MODELS), pl.col("score").round(2))
        .sort("model", "duration", "layer", "language", "metric")
    )


def inline_csv(df: pl.DataFrame) -> alt.InlineData:
    return alt.InlineData(values=df.write_csv(), format=alt.CsvDataFormat(type="csv"))


def plot_across_layers(scores: pl.DataFrame) -> alt.LayerChart:
    metric_select, language_select, finetuning_select, legend_select = common_selectors()
    share_y = alt.param(name="share_y", bind=alt.binding_checkbox(name="Shared y-axis"), value=False)

    base = alt.Chart(inline_csv(scores.drop("best_layer")))
    bg_encoding = base.mark_point(opacity=0, size=0).encode(y=alt.Y("score:Q", scale=alt.Scale(zero=False)))
    bg_shared = bg_encoding.transform_filter(metric_select).transform_filter("share_y")
    bg_local = (
        bg_encoding.transform_filter(metric_select).transform_filter(language_select).transform_filter("!share_y")
    )
    fg, rules, points = _baseline_layers(base, [metric_select, finetuning_select, language_select], legend_select)
    title_expr = f"({METRIC_NAME_EXPR}) + ' — ' + ({SCORES_LANG_NAME_EXPR}) + ' — ' + ({FT_NAME_EXPR})"
    return (
        alt.layer(bg_shared, bg_local, fg, points, rules)
        .add_params(share_y, finetuning_select, metric_select, language_select)
        .properties(width="container", height=300, title=alt.Title(text={"expr": title_expr}))
    )


def plot_best_layer(scores: pl.DataFrame) -> alt.LayerChart:
    metric_select, language_select, _, legend_select = common_selectors()
    share_y = alt.param(name="share_y", bind=alt.binding_checkbox(name="Shared y-axis"), value=False)

    base = alt.Chart(inline_csv(scores.filter("best_layer").drop("best_layer")))
    bg_encoding = base.mark_point(opacity=0, size=0).encode(y=alt.Y("score:Q", scale=alt.Scale(zero=False)))
    bg_shared = bg_encoding.transform_filter(metric_select).transform_filter("share_y")
    bg_local = (
        bg_encoding.transform_filter(metric_select).transform_filter(language_select).transform_filter("!share_y")
    )
    fg, rules, points = _baseline_layers(base, [metric_select, language_select], legend_select, x=_DURATION_X)
    title_expr = f"({METRIC_NAME_EXPR}) + ' — ' + ({SCORES_LANG_NAME_EXPR})"
    return (
        alt.layer(bg_shared, bg_local, fg, points, rules)
        .add_params(share_y, metric_select, language_select)
        .properties(width="container", height=200, title=alt.Title(text={"expr": title_expr}))
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
        .join(pl.read_ndjson(root_manifests / "speakers.jsonl").drop("split"), on=["speaker", "language"])
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


HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <link rel="stylesheet" href="https://fonts.googleapis.com/css?family=Inter:400,500&display=fallback">
  <link rel="stylesheet" href="{root}/stylesheets/vega.css">
  <script src="https://cdn.jsdelivr.net/npm/vega@{vega}"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-lite@{vegalite}"></script>
  <script src="https://cdn.jsdelivr.net/npm/vega-embed@{vegaembed}"></script>
</head>
<body>
  <div id="vis"></div>
  <script>window.spec = {spec};</script>
  <script src="{root}/javascripts/vega-figure.js"></script>
</body>
</html>
"""


def to_html(chart: alt.Chart | alt.LayerChart | alt.FacetChart, root: str = "..") -> str:
    """Standalone HTML of a figure, styled like the documentation (see docs/javascripts/vega-figure.js)."""
    return HTML_TEMPLATE.format(
        root=root,
        vega=alt.VEGA_VERSION,
        vegalite=alt.VEGALITE_VERSION,
        vegaembed=alt.VEGAEMBED_VERSION,
        spec=chart.to_json(indent=None),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path, help="Path to the benchmark dataset")
    parser.add_argument("artifacts", type=Path, help="Path to the artifacts dataset")
    parser.add_argument("destination", type=Path, help="Path to the assets directory in docs")
    args = parser.parse_args()
    scores = baseline_scores(args.artifacts)

    def write(chart: alt.Chart | alt.LayerChart | alt.FacetChart, name: str) -> None:
        (args.destination / name).write_text(to_html(chart), encoding="utf-8")

    write(plot_across_layers(scores), "baseline_across_layers.html")
    write(plot_best_layer(scores), "baseline_best_layer.html")
    write(plot_datasets_stats(args.dataset / "manifest"), "dataset_stats.html")
    write(plot_speakers_stats(args.dataset / "manifest"), "speaker_stats.html")
    write(plot_phone_distribution(args.dataset / "alignment"), "phone_distribution.html")
