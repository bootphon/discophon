# %%
from pathlib import Path

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from discophon.core.builder import (
    build_items,
    build_wolof_alignments,
    check_consecutive,
    read_lexicon,
    read_old_alignments,
    read_text,
)

PHONOLOGY_WOLOF = {
    "@": "ə",  # central mid vowel
    "E": "ɛ",  # open-mid front vowel
    "J": "ɲ",  # palatal nasal
    "Jj": "ᶮɟ",  # prenasalized palatal plosive
    "N": "ŋ",  # velar nasal
    "Ng": "ᵑɡ",  # prenasalized velar plosive
    "Nq": "ɴq",  # prenasalized uvular plosive
    "O": "ɔ",  # open-mid back vowel
    "SIL": "SIL",  # silence
    "a": "a",  # open central vowel
    "b": "b",  # voiced labial plosive
    "c": "c",  # voiceless palatal plosive
    "d": "d",  # voiced alveolar plosive
    "e": "e",  # close-mid front vowel
    "f": "f",  # labial fricative
    "g": "g",  # voiced velar plosive
    "h": None,  # should have been removed. No h at all in dictionnary, but it appears in the transcriptions used for the n-gram models that did the lexicon extraction. Maybe used as a glottal stop? In any case, remove
    "i": "i",  # close front vowel
    "j": "ɟ",  # voiced palatal plosive # same as Jj
    "k": "k",  # voiceless velar plosive
    "l": "l",  # alveolar approximant
    "m": "m",  # labial nasal
    "mb": "ᵐb",  # prenasalized labial plosive
    "n": "n",  # alveolar nasal
    "nd": "ⁿd",  # prenasalized alveolar plosive
    "o": "o",  # close-mid back vowel
    "p": "p",  # voiceless labial plosive
    "q": "q",  # voiceless uvular plosive
    "r": "r",  # alveolar trill
    "s": "s",  # alveolar fricative
    "t": "t",  # voiceless alveolar plosive
    "u": "u",  # close back vowel
    "w": "w",  # labial approximant
    "x": "x",  # velar-uvular fricative
    "y": "j",  # palatal approximant
}
MAY_BE_LONG = {"i", "e", "ɛ", "a", "o", "u", "ɔ"}

root = Path(
    "/store/projects/zerospeech/archive/2017/archives/LANG1/output_corpus_creation_wolof/models/alignment_phone/"
)

SPLIT = "dev"
MANIFEST = pl.read_csv(f"/store/projects/phoneme_discovery/benchmark/manifests/manifest-wol-{SPLIT}.csv")
FILES = set(MANIFEST["fileid"].unique().to_list())
ALIGNMENTS = read_old_alignments(root / "alignment.txt").filter(pl.col("fileid").is_in(FILES))
LEXICON = read_lexicon(root / "recipe/data/local/align/lexicon.txt")
TEXT = read_text(root / "recipe/data/align/text")

df = build_wolof_alignments(LEXICON, TEXT, ALIGNMENTS, PHONOLOGY_WOLOF, MAY_BE_LONG).rename(
    {"fileid": "#file", "phone": "#phone"}
)

plt.figure(figsize=(8, 4))
sns.barplot(df["#phone"].value_counts().sort("count"), x="#phone", y="count")
plt.title("Phone distribution in Wolof alignment")
plt.tight_layout()
plt.yscale("log")
plt.show()


check_consecutive(df)
# df.write_csv(f"/store/projects/phoneme_discovery/benchmark/alignment/alignment-wol-{SPLIT}.txt", separator=" ")
phoneme, triphone = build_items(
    df.with_columns(("WOL_" + pl.col("#file").str.split("_").list.get(1)).alias("speaker"))
)
# phoneme.write_csv(f"/store/projects/phoneme_discovery/benchmark/item/phoneme-wolof-{SPLIT}.item", separator=" ")
# triphone.write_csv(f"/store/projects/phoneme_discovery/benchmark/item/triphone-wolof-{SPLIT}.item", separator=" ")
# %%
df["#phone"].value_counts().sort("count")
# %%
