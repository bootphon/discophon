# %%
import json
from pathlib import Path

import altair as alt
import polars as pl
from discophon.core import Language
from discophon.core.data import read_gold_annotations_as_dataframe

phonolgy = json.loads(Path("phonolgy.json").read_text(encoding="utf-8"))
# phonolgy = json.loads(Path("./builder/src/discophon/builder/rules/rules.json").read_text(encoding="utf-8"))

root = Path("/store/projects/phoneme_discovery/benchmark/alignment/")
df = {
    str(lang): pl.concat(
        (
            # read_gold_annotations_as_dataframe(root / f"alignment-{lang}-dev.txt"),
            read_gold_annotations_as_dataframe(root / f"alignment-{lang}-test.txt"),
        )
    ).filter(pl.col("#phone") != "SIL")
    # .with_columns(pl.col("#phone").replace_strict(phonolgy[str(lang)]))
    for lang in sorted(Language)
    if (root / f"alignment-{lang}-test.txt").is_file()
}

counts = {lang: x["#phone"].value_counts() for lang, x in df.items()}


def make_barchart(counts: pl.DataFrame) -> alt.Chart:
    return (
        alt.Chart(counts)
        .mark_bar()
        .encode(
            x=alt.X("#phone:N", title="Phoneme", sort="-y").axis(labelAngle=0),
            y=alt.Y("count:Q", sort="-y", title="Count", scale=alt.Scale(type="linear")),
            tooltip=["#phone", "count"],
        )
    )


chart = alt.vconcat(
    *[make_barchart(count).properties(title=f"{code}, num phonemes: {count.height}") for code, count in counts.items()]
).resolve_scale(y="independent")
chart
# chart.save("phoneme_distributions.html")
# %%

files = {
    "jpn": (
        df["jpn"]
        .filter(pl.col("#phone").is_in(counts["jpn"].filter(pl.col("count") < 20)["#phone"].implode()))["#file"]
        .to_list()
    )
}

# %%
from fastabx.dataset import read_labels


def remove(files: dict[str, list[str]], lang: str) -> None:
    for split in ["dev", "test"]:
        align = read_gold_annotations_as_dataframe(root / f"alignment-{lang}-{split}.txt")
        filtered = align.filter(~pl.col("#file").is_in(files[lang]))
        filtered.write_csv(root / f"alignment-{lang}-{split}.txt", separator=" ")
        for unit in ["triphone", "phoneme"]:
            item = read_labels(root / f"../item/{unit}-{lang}-{split}.item", "#file", "onset", "offset")
            filtered_item = item.filter(~item["#file"].is_in(files[lang]))
            filtered_item.write_csv(root / f"../item/{unit}-{lang}-{split}.item", separator=" ")


remove(files, "jpn")

# for lang in to_remove:
#     remove(to_remove, lang)
# %%
x = df["fra"]["#phone"].unique().sort().to_list()
dict(zip(x, x))
# %%
len(to_remove)
# %%
for f in files["jpn"]:
    p = Path(f"/store/projects/phoneme_discovery/benchmark/audio/jpn/all/{f}.wav")
    p.unlink()
    if (p.parent / f"../dev/{p.name}").is_symlink():
        (p.parent / f"../dev/{p.name}").unlink()
    if (p.parent / f"../test/{p.name}").is_symlink():
        (p.parent / f"../test/{p.name}").unlink()

# %%
