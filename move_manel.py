# %%
from pathlib import Path
from shutil import copy2

from discophon.core import Language

root_src = Path("/mnt/legacy_nas1/projects/phoneme_discovery_benchmark/splits_improved")
dest = Path("/mnt/legacy_nas1/projects/phoneme_discovery/benchmark")

for code in ("sw", "ta", "th", "tr", "uk", "zh-CN"):
    lang = Language.from_commonvoice(code)
    src = root_src / "dev-lang" / code if lang.split == "dev" else root_src / "test-lang" / code

    # Copy alignments
    for split in ["dev", "test"]:
        align = src / "alignment" / f"{code}-{split}.align"
        copy2(align, dest / "alignment" / f"alignment-{lang!s}-{split}.txt")

    # Copy item
    for split in ["dev", "test"]:
        for mine, manel in [("phoneme", "phonebase"), ("triphone", "triphone")]:
            item = src / "item" / f"{code}-{manel}-{split}.item"
            copy2(item, dest / "item" / f"{mine}-{lang!s}-{split}.item")
# %%
