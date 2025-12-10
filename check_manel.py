# %%%
from datetime import timedelta
from pathlib import Path

import altair as alt
import polars as pl

alt.data_transformers.enable("vegafusion")


def read_manel_manifest(path: str | Path) -> pl.DataFrame:
    return (
        pl.read_csv(path)
        .rename({"speaker_id": "client_id", "duration": "duration", "filename": "path"})
        .with_columns(
            pl.col("gender")
            .fill_null("unknown")
            .replace_strict({"female_feminine": "F", "male_masculine": "M", "unknown": "<unk>"})
            .alias("gender")
        )
        .select(["client_id", "path", "gender", "duration"])
        .group_by("client_id")
        .agg(pl.col("gender").first(), pl.col("duration").sum())
    )


def make_chart(df: pl.DataFrame, name: str) -> alt.LayerChart:
    duration = timedelta(seconds=int(df["duration"].sum()))
    n_male, n_female = len(df.filter(pl.col("gender") == "M")), len(df.filter(pl.col("gender") == "F"))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("client_id:N", title="Speaker").sort("-y"),
            y=alt.Y("duration:Q", title="Duration (seconds)"),
            color=alt.Color(
                "gender:N",
                title="Gender",
                scale=alt.Scale(domain=["F", "M", "<unk>"], range=["#ff69b4", "#1f77b4", "#d3d3d3"]),
            ),
            tooltip=["client_id", "gender", "duration"],
        )
        .properties(title=f"{name}. Total duration: {duration}. Speakers: {n_male} M, {n_female} F")
    ) + (alt.Chart(pl.DataFrame({"y": [7_200 / 20]})).mark_rule(color="black").encode(y="y:Q"))


root = Path("/store/projects/phoneme_discovery_benchmark/splits_improved/")
chart = None
for lang in ["sw", "ta", "th", "tr", "uk", "es", "eu", "ja", "zh-CN"]:
    kind = "dev-lang" if lang in {"sw", "ta", "th", "tr", "uk"} else "test-lang"
    manifest_dev = read_manel_manifest(root / f"{kind}/{lang}/{lang}-dev.csv")
    manifest_test = read_manel_manifest(root / f"{kind}/{lang}/{lang}-test.csv")
    dev = make_chart(manifest_dev, f"{lang} - dev")
    test = make_chart(manifest_test, f"{lang} - test")
    new_chart = dev | test
    chart = new_chart if chart is None else chart & new_chart

# manifest = read_manel_manifest("./dev_correct.csv")
# chart = make_chart(manifest, "dev_correct")
# chart.save("dev_correct.html")

# %%
chart.save("splits_improved.pdf")
# %%
chart.save("splits_improved.png")

# %%
