# %%
import json
from pathlib import Path

import polars as pl

jpn = json.loads(Path("./builder/src/discophon/builder/rules/rules.json").read_text(encoding="utf-8"))["jpn"]
item = pl.read_csv("/store/projects/phoneme_discovery/benchmark/item/triphone-jpn-dev.item", separator=" ")

# %%
item.filter(~pl.col("#phone").is_in(list(jpn.values())))["#phone"].unique()
# %%
jpn.values()
# %%
list(jpn.values())
# %%
