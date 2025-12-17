# %%
from pathlib import Path

import polars as pl

paths = {
    # "cmn": Path("/store/data/raw_data/commonvoice/cv17/cv17_zh-CN"),
    # "eus": Path("/store/data/raw_data/commonvoice/cv-corpus-22.0-2025-06-20/eu"),
    "jpn": Path("/store/data/raw_data/commonvoice/cv-corpus-22.0-2025-06-20/ja"),
    # "swa": Path("/mnt/legacy_nas1/data/raw_data/commonvoice/cv23/audio/swahili"),
    # "tam": Path("/mnt/legacy_nas1/data/raw_data/commonvoice/cv23/audio/tamil"),
    # "tha": Path("/mnt/legacy_nas1/data/raw_data/commonvoice/cv23/audio/thai"),
    # "tur": Path("/mnt/legacy_nas1/data/raw_data/commonvoice/cv23/audio/turkish"),
    # "ukr": Path("/mnt/legacy_nas1/data/raw_data/commonvoice/cv23/audio/ukranian"),
}
validated = {
    lang: pl.read_csv(path / "validated.tsv", separator="\t", quote_char=None)
    .with_columns(pl.col("gender").fill_null("<unk>"))
    .with_columns(
        pl.col("gender").replace_strict(
            {
                "male_masculine": "M",
                "female_feminine": "F",
                "<unk>": "<unk>",
                "do_not_wish_to_say": "<unk>",
                "transgender": "O",
                "non-binary": "O",
            }
        )
    )
    .group_by("client_id")
    .agg("gender")
    .with_columns(pl.col("gender").list.unique())
    .rename({"client_id": "speaker"})
    for lang, path in paths.items()
}

# %%

speakers = {
    lang: (
        pl.concat(
            [
                pl.read_csv(p)
                .select("speaker")
                .unique()
                .with_columns(pl.lit(p.stem.removeprefix(f"manifest-{lang}-")).alias("split"))
                for p in Path("/store/projects/phoneme_discovery/benchmark/manifest").glob(f"manifest-{lang}-*.csv")
            ]
        )
        .group_by("speaker")
        .agg(pl.col("split"))
        .sort("speaker")
        .with_columns(pl.lit(lang).alias("language"))
    )
    for lang in paths
}

merged = (
    pl.concat(
        [
            speakers[lang]
            .join(validated[lang], on="speaker", how="left")
            .select("language", "speaker", "gender", "split")
            for lang in paths
        ]
    )
    .with_columns(
        pl.when(pl.col("gender").list.len() == 1)
        .then(pl.col("gender").list.first())
        .otherwise(
            pl.when(pl.col("gender").list.contains("O"))
            .then(pl.lit("O"))
            .otherwise(
                pl.when(pl.col("gender").list.contains("F") & pl.col("gender").list.contains("M"))
                .then(pl.lit("<unk>"))
                .otherwise(pl.when(pl.col("gender").list.contains("F")).then(pl.lit("F")).otherwise(pl.lit("M")))
            )
        )
        .alias("gender")
    )
    .with_columns(
        pl.when(
            pl.col("speaker").is_in(
                [
                    "3da9aa75aba1f0a7124f2284342a9f98690c0037ce0390fb77c1bec83c2d340a4831ea10a27676edeb7c26ddb590e560d6b3631c004e19d6c2d3e6d3c50c4a3a",
                    "9bf3876a56bbc5427b675ddd8bf6230b07b6337d977c92919285e57160c97f2388077f44c8420e5d8076ac3ad3bd2dd69631dd39d6e9edcd7b1f3049091e29d2",
                ]
            )
        )
        .then(pl.lit("F"))
        .otherwise("gender")
        .alias("gender")
    )
)
# %%
merged.filter(~pl.col("gender").is_in(["M", "F"]), pl.col("split").list.contains("dev"))

# %%
existing = pl.read_ndjson("/store/projects/phoneme_discovery/benchmark/manifest/speakers.jsonl")
pl.concat((existing, merged)).sort("language", "speaker").write_ndjson(
    "/store/projects/phoneme_discovery/benchmark/manifest/speakers.jsonl"
)
# %%
