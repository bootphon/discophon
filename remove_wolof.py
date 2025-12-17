# %%
from decimal import Decimal

import polars as pl

for name in ["triphone-wol-dev.item", "phoneme-wol-dev.item", "triphone-wol-test.item", "phoneme-wol-test.item"]:
    path = f"/store/projects/phoneme_discovery/benchmark/item/{name}"
    (
        pl.read_csv(
            path,
            separator=" ",
            schema_overrides={"onset": pl.Decimal(20, 6), "offset": pl.Decimal(20, 6)},
        )
        .with_columns(pl.col("onset") - Decimal("0.0125"), pl.col("offset") - Decimal("0.0125"))
        .write_csv(path, separator=" ")
    )

# %%
df
# %%
