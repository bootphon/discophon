# %%
import enum

import matplotlib.pyplot as plt
import polars as pl
import seaborn as sns
from discophon.core.builder import pad_decimal_zeros


class Special(enum.StrEnum):
    SIL = enum.auto()
    SPN = enum.auto()
    TO_REMOVE = enum.auto()


SPLIT = "test"

MANIFESTS = {
    "dev": pl.read_csv("/store/projects/phoneme_discovery/benchmark/manifests/manifest-wol-dev.csv"),
    "test": pl.read_csv("/store/projects/phoneme_discovery/benchmark/manifests/manifest-wol-test.csv"),
}
FILES = {split: set(df["fileid"].unique().to_list()) for split, df in MANIFESTS.items()}

ALIGNMENTS = (
    pl.read_csv(
        "/store/projects/zerospeech/archive/2017/archives/LANG1/output_corpus_creation_wolof/models/alignment_phone/alignment.txt",
        has_header=False,
        separator=" ",
        new_columns=["fileid", "onset", "offset", "proba", "phone"],
        schema_overrides={"onset": pl.String, "offset": pl.String},
    )
    .filter(pl.col("fileid").is_in(FILES[SPLIT]))
    .drop("proba")
    .with_columns(pad_decimal_zeros("onset", 6), pad_decimal_zeros("offset", 6))
)

LEXICON = (
    pl.read_csv(
        "/store/projects/zerospeech/archive/2017/archives/LANG1/output_corpus_creation_wolof/models/alignment_phone/recipe/data/local/align/lexicon.txt",
        has_header=False,
        new_columns=["full"],
    )
    .with_columns(
        pl.col("full").str.split(" ").list.get(0).alias("word"),
        pl.col("full").str.split(" ").list.slice(1).alias("phones"),
    )
    .select("word", "phones")
)

TEXT = (
    pl.read_csv(
        "/store/projects/zerospeech/archive/2017/archives/LANG1/output_corpus_creation_wolof/models/alignment_phone/recipe/data/align/text",
        has_header=False,
        new_columns=["full"],
    )
    .with_columns(
        pl.col("full").str.split(" ").list.get(0).alias("fileid"),
        pl.col("full").str.split(" ").list.slice(1).alias("words"),
    )
    .select("fileid", "words")
)

PHONOLOGY_WOLOF = {
    "@": "ə",  # central mid vowel
    "E": "ɛ",  # open-mid front vowel
    "J": "ɲ",  # palatal nasal
    "Jj": "ɟ",  # voiced palatal plosive # same as j, but could be instead mapped to ᶮɟ (prenasalized palatal plosive)
    "N": "ŋ",  # velar nasal
    "Ng": "ᵑɡ",  # prenasalized velar plosive
    "Nq": Special.TO_REMOVE,  # should have been separated in two (maybe?)
    "O": "ɔ",  # open-mid back vowel
    "SIL": Special.SIL,  # silence
    "a": "a",  # open central vowel
    "b": "b",  # voiced labial plosive
    "c": "c",  # voiceless palatal plosive
    "d": "d",  # voiced alveolar plosive
    "e": "e",  # close-mid front vowel
    "f": "f",  # labial fricative
    "g": "g",  # voiced velar plosive
    "h": Special.TO_REMOVE,  # should have been removed. No h at all in dictionnary, but it appears in the transcriptions used for the n-gram models that did the lexicon extraction. Maybe used as a glottal stop? In any case, remove
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

plt.figure(figsize=(8, 4))
sns.barplot(ALIGNMENTS["phone"].value_counts().sort("count"), x="phone", y="count")
plt.title("Phone distribution in Wolof alignment")
plt.tight_layout()
plt.yscale("log")
plt.show()

# %%
ALIGNMENTS
# %%
TEXT
# %%
to_remove = ALIGNMENTS.filter((pl.col("phone") == "Nq") | (pl.col("phone") == "h"))["fileid"].unique().to_list()

# %%
MANIFESTS["test"].filter(~pl.col("fileid").is_in(to_remove)).write_csv(
    "/store/projects/phoneme_discovery/benchmark/manifests/manifest-wol-test.csv"
)
# %%
