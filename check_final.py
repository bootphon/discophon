# %%
from datetime import timedelta
from pathlib import Path

import altair as alt
import polars as pl

alt.data_transformers.enable("vegafusion")

ROOT = Path("/store/projects/phoneme_discovery/benchmark/")
LANGUAGES = ["cmn", "eus", "eng", "fra", "jpn", "swa", "tam", "tha", "tur", "ukr", "wol"]


def read_all() -> pl.DataFrame:
    manifests = []
    for lang in LANGUAGES:
        print(lang)
        manifests.extend(
            [
                pl.read_csv(
                    ROOT / f"manifest/manifest-{lang}-{split}.csv", schema_overrides={"speaker": pl.String}
                ).with_columns(pl.lit(lang).alias("language"), pl.lit(split).alias("split"))
                for split in ["dev", "test", "train-10min", "train-1h", "train-10h"]
                if (ROOT / f"manifest/manifest-{lang}-{split}.csv").is_file()
            ]
        )
    df = pl.concat(manifests)
    gender = pl.read_ndjson(ROOT / "manifest/speakers.jsonl").select(["speaker", "gender"])
    out = df.join(gender, on="speaker", how="left")
    # assert len(df) == len(out)
    return out


def make_chart(df: pl.DataFrame, name: str) -> alt.LayerChart:
    duration = timedelta(seconds=int(df["duration"].sum()))
    n_male, n_female = len(df.filter(pl.col("gender") == "M")), len(df.filter(pl.col("gender") == "F"))
    return (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X("speaker:N", title="Speaker").sort("-y"),
            y=alt.Y("duration:Q", title="Duration (seconds)"),
            color=alt.Color(
                "gender:N",
                title="Gender",
                scale=alt.Scale(domain=["F", "M", "<unk>"], range=["#ff69b4", "#1f77b4", "#d3d3d3"]),
            ),
            tooltip=["speaker", "gender", "duration"],
        )
        .properties(title=f"{name}. Total duration: {duration}. Speakers: {n_male} M, {n_female} F")
    ) + (alt.Chart(pl.DataFrame({"y": [7_200 / 20]})).mark_rule(color="black").encode(y="y:Q"))


df = read_all().with_columns((pl.col("num_samples") / 16000).alias("duration"))

#%%
charts = []
for lang in LANGUAGES:
    for split in ["dev", "test"]:
        subdf = df.filter((pl.col("language") == lang) & (pl.col("split") == split))
        subdf = subdf.group_by("speaker").agg(pl.col("gender").first(), pl.col("duration").sum())
        chart = make_chart(subdf, f"{lang} - {split}")
        charts.append(chart)
chart = alt.vconcat(*charts)
chart

# %%
