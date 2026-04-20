import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Literal

__all__ = ["Language", "get_language"]

type Sonority = Literal[
    "fricative",
    "affricate",
    "plosive",
    "vibrant",
    "nasal",
    "approximant",
    "monophthong",
    "diphthong",
]


@cache
def _load_asset(name: str) -> dict:
    return json.loads((resources.files(__package__) / f"assets/{name}.json").read_text(encoding="utf-8"))


def load_sonority() -> dict[str, Sonority]:
    return _load_asset("sonority")


def load_tipa() -> dict[str, str]:
    return _load_asset("tipa")


def load_phonemes() -> dict[str, list[str]]:
    return _load_asset("phonemes")


@dataclass(frozen=True)
class Language:
    """The underlying representation of a language.

    Parameters:
        name: Name of the language.
        iso_639_3: Its ISO 639-3 code.
        split: Which split it belongs to in the benchmark.
        n_phonemes: The number of phoneme categories considered.
    """

    name: str
    iso_639_3: str
    split: Literal["dev", "test"]
    n_phonemes: int

    def __post_init__(self) -> None:
        if self.n_phonemes != len(self.phonemes):
            raise ValueError(f"Internal failure: {self.n_phonemes=} != {len(self.phonemes)=} for {self.name}.")

    @property
    def phonemes(self) -> list[str]:
        """The phonemes of this language."""
        return load_phonemes()[self.iso_639_3]


def get_language(n: str | Language, /) -> Language:  # noqa: C901, PLR0911, PLR0912
    """Return the language corresponding to this string."""
    if isinstance(n, Language):
        return n
    match n.lower():
        case "german" | "deu":
            return Language(name="German", iso_639_3="deu", split="dev", n_phonemes=41)
        case "swahili" | "swa" | "sw":
            return Language(name="Swahili", iso_639_3="swa", split="dev", n_phonemes=29)
        case "tamil" | "tam" | "ta":
            return Language(name="Tamil", iso_639_3="tam", split="dev", n_phonemes=29)
        case "thai" | "tha" | "th":
            return Language(name="Thai", iso_639_3="tha", split="dev", n_phonemes=40)
        case "turkish" | "tur" | "tr":
            return Language(name="Turkish", iso_639_3="tur", split="dev", n_phonemes=27)
        case "ukrainian" | "ukr" | "uk":
            return Language(name="Ukrainian", iso_639_3="ukr", split="dev", n_phonemes=35)
        case "mandarin chinese" | "mandarin" | "chinese" | "cmn" | "zh-CN":
            return Language(name="Mandarin Chinese", iso_639_3="cmn", split="test", n_phonemes=42)
        case "english" | "eng":
            return Language(name="English", iso_639_3="eng", split="test", n_phonemes=39)
        case "basque" | "eus" | "eu":
            return Language(name="Basque", iso_639_3="eus", split="test", n_phonemes=29)
        case "french" | "fra":
            return Language(name="French", iso_639_3="fra", split="test", n_phonemes=34)
        case "japanese" | "jpn" | "ja":
            return Language(name="Japanese", iso_639_3="jpn", split="test", n_phonemes=42)
        case "wolof" | "wol":
            return Language(name="Wolof", iso_639_3="wol", split="test", n_phonemes=39)
    raise ValueError(f"Unknown language '{n}'")


type TupleOfSixLanguages = tuple[Language, Language, Language, Language, Language, Language]


def dev_languages() -> TupleOfSixLanguages:
    return (
        get_language("deu"),
        get_language("swa"),
        get_language("tam"),
        get_language("tha"),
        get_language("tur"),
        get_language("ukr"),
    )


def test_languages() -> TupleOfSixLanguages:
    return (
        get_language("cmn"),
        get_language("eng"),
        get_language("eus"),
        get_language("fra"),
        get_language("jpn"),
        get_language("wol"),
    )


def all_languages() -> tuple[Language, ...]:
    return dev_languages() + test_languages()


def languages_in_split(s: Literal["dev", "test"], /) -> TupleOfSixLanguages:
    match s:
        case "dev":
            return dev_languages()
        case "test":
            return test_languages()
    raise ValueError(f"Unknown split '{s}'")


def commonvoice_languages() -> tuple[Language, ...]:
    return (
        get_language("swa"),
        get_language("tam"),
        get_language("tha"),
        get_language("tur"),
        get_language("ukr"),
        get_language("cmn"),
        get_language("eus"),
        get_language("jpn"),
    )


ISO6393_TO_CV = {
    "swa": "sw",
    "tam": "ta",
    "tha": "th",
    "tur": "tr",
    "ukr": "uk",
    "cmn": "zh-CN",
    "eus": "eu",
    "jpn": "jpn",
}
