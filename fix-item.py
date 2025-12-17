import json
from pathlib import Path

import polars as pl
from fastabx.dataset import read_labels


def read_alignment(source: str | Path) -> pl.DataFrame:
    df = pl.read_csv(
        source,
        separator=" ",
        columns=["#file", "onset", "offset", "#phone"],
        schema_overrides=[pl.String] * 4,
    )
    return df.with_columns(
        df["onset"].str.to_decimal(inference_length=len(df)),
        df["offset"].str.to_decimal(inference_length=len(df)),
    ).sort("#file", "onset")


def read_item(path: str | Path) -> pl.DataFrame:
    return read_labels(path, "#file", "onset", "offset")


if __name__ == "__main__":
    root = Path("/store/projects/phoneme_discovery/benchmark/alignment/")
    phonology = json.loads(Path("phonolgy.json").read_text(encoding="utf-8"))
    for lang, mapping in phonology.items():
        m = mapping | {"SIL": "SIL"}
        for split in ["dev", "test"]:
            align = root / f"alignment-{lang}-{split}.txt"
            read_alignment(align).with_columns(pl.col("#phone").replace_strict(m)).write_csv(align, separator=" ")
            item = root.parent / "item" / f"triphone-{lang}-{split}.item"
            read_item(item).with_columns(
                pl.col("#phone").replace_strict(m),
                pl.col("next-phone").replace_strict(m),
                pl.col("prev-phone").replace_strict(m),
            ).write_csv(item, separator=" ")
