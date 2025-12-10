# %%
import altair as alt
import polars as pl

df = (
    pl.read_csv("/store/projects/phoneme_discovery/benchmark/manifest/manifest-wol-test.csv")
    .join(pl.read_ndjson("/store/projects/phoneme_discovery/benchmark/manifest/speakers.jsonl"), on="speaker")
    .with_columns((pl.col("num_samples") / 16_000).alias("duration"))
)

# %%
duration = df["duration"].sum()
n_male, n_female = len(df.filter(pl.col("gender") == "M")), len(df.filter(pl.col("gender") == "F"))
chart = (
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
    .properties(title=f"Total duration: {duration:.2f} seconds. Speakers: {n_male} M, {n_female} F")
) + (alt.Chart(pl.DataFrame({"y": [7_200 / 20]})).mark_rule(color="black").encode(y="y:Q"))

# %%
chart
# %%
df["speaker"].value_counts()
# %%
df.join(x, on="speaker")
# %%
df["duration"] / 7200
# %%
(duration - 7200) / 60
# %%
