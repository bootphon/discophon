import json
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Literal


@cache
def load_phonology() -> dict[str, list[str]]:
    return json.loads((resources.files("discophon") / "core/assets/phonology.json").read_text(encoding="utf-8"))


@cache
def load_tipa() -> dict[str, str]:
    return json.loads((resources.files("discophon") / "core/assets/tipa.json").read_text(encoding="utf-8"))


@dataclass(frozen=True)
class Language:
    name: str
    iso_639_3: str
    split: str
    n_phonemes: int

    @property
    def phonology(self) -> list[str]:
        phonemes = load_phonology()[self.iso_639_3]
        assert len(phonemes) == self.n_phonemes
        return phonemes


def language(n: str, /) -> Language:  # noqa: C901, PLR0911
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
    return language("deu"), language("swa"), language("tam"), language("tha"), language("tur"), language("ukr")


def test_languages() -> TupleOfSixLanguages:
    return language("cmn"), language("eng"), language("eus"), language("fra"), language("jpn"), language("wol")


def languages_in_split(s: Literal["dev", "test"], /) -> TupleOfSixLanguages:
    match s:
        case "dev":
            return dev_languages()
        case "test":
            return test_languages()
    raise ValueError(f"Unknown split '{s}'")


def commonvoice_languages() -> tuple[Language, ...]:
    return (
        language("swa"),
        language("tam"),
        language("tha"),
        language("tur"),
        language("ukr"),
        language("cmn"),
        language("eus"),
        language("jpn"),
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
