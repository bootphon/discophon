from enum import StrEnum
from typing import Literal

COMMONVOICE_TO_ISO639_3 = {
    "eu": "eus",
    "ja": "jpn",
    "sw": "swa",
    "ta": "tam",
    "th": "tha",
    "tr": "tur",
    "uk": "ukr",
    "zh-CN": "cmn",
}


class Language(StrEnum):
    # Dev languages
    DEU = "deu"  # German
    SWA = "swa"  # Swahili
    TAM = "tam"  # Tamil
    THA = "tha"  # Thai
    TUR = "tur"  # Turkish
    UKR = "ukr"  # Ukrainian

    # Test languages
    CMN = "cmn"  # Mandarin Chinese
    ENG = "eng"  # English
    EUS = "eus"  # Basque
    FRA = "fra"  # French
    JPN = "jpn"  # Japanese
    WOL = "wol"  # Wolof

    @classmethod
    def from_commonvoice(cls, code: str) -> "Language":
        return cls(COMMONVOICE_TO_ISO639_3[code])

    @property
    def split(self) -> Literal["dev", "test"]:
        if self in {Language.UKR, Language.THA, Language.TUR, Language.SWA, Language.TAM, Language.DEU}:
            return "dev"
        return "test"
