# %%
import json
from pathlib import Path

import polars as pl

phonology = json.loads(Path("phonolgy.json").read_text(encoding="utf-8"))["eus"] | {"SIL": "SIL"}
split = "test"
manifest = pl.read_csv(f"/mnt/legacy_nas1/projects/phoneme_discovery/benchmark/manifest/manifest-eus-{split}.csv")
alignment = (
    pl.read_csv(
        f"/mnt/legacy_nas1/projects/phoneme_discovery/benchmark/alignment/alignment-eus-{split}.txt",
        separator=" ",
        schema_overrides=[pl.String] * 4,
    )
    .filter(pl.col("#file").is_in(manifest["fileid"].implode()))
    .with_columns(pl.col("#phone").replace_strict(phonology))
)

phoneme = (
    pl.read_csv(
        f"/mnt/legacy_nas1/projects/phoneme_discovery/benchmark/item/phoneme-eus-{split}.item",
        separator=" ",
        schema_overrides=[pl.String] * 7,
    )
    .filter(pl.col("#file").is_in(manifest["fileid"].implode()))
    .with_columns(
        pl.col("#phone").replace_strict(phonology),
        pl.col("next-phone").replace_strict(phonology),
        pl.col("prev-phone").replace_strict(phonology),
    )
)

triphone = (
    pl.read_csv(
        f"/mnt/legacy_nas1/projects/phoneme_discovery/benchmark/item/triphone-eus-{split}.item",
        separator=" ",
        schema_overrides=[pl.String] * 7,
    )
    .filter(pl.col("#file").is_in(manifest["fileid"].implode()))
    .with_columns(
        pl.col("#phone").replace_strict(phonology),
        pl.col("next-phone").replace_strict(phonology),
        pl.col("prev-phone").replace_strict(phonology),
    )
)
alignment.write_csv(
    f"/mnt/legacy_nas1/projects/phoneme_discovery/benchmark/alignment/alignment-eus-{split}.txt", separator=" "
)
phoneme.write_csv(
    f"/mnt/legacy_nas1/projects/phoneme_discovery/benchmark/item/phoneme-eus-{split}.item", separator=" "
)
triphone.write_csv(
    f"/mnt/legacy_nas1/projects/phoneme_discovery/benchmark/item/triphone-eus-{split}.item", separator=" "
)
# %%
x = set(alignment["#file"].to_list())
y = set(manifest["fileid"].to_list())
# %%
len(x - y)
# %%
len(y - x)
# %%
alignment
# %%
phoneme
# %%
