# %%
from discophon.core.data import read_gold_annotations_as_dataframe

df = read_gold_annotations_as_dataframe(
    "/store/projects/phoneme_discovery_benchmark/splits_improved/dev-lang/uk/alignment/uk-test.align"
)

# %%
d = df["#phone"].value_counts().sort("#phone").to_dict()
dict(zip(d["#phone"], d["count"]))
# %%
