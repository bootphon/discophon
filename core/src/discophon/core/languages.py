from enum import StrEnum
from typing import Literal


class Language(StrEnum):
    # Dev languages
    UKR = "ukr"  # Ukrainian
    THA = "tha"  # Thai
    TUR = "tur"  # Turkish
    SWA = "swa"  # Swahili
    TAM = "tam"  # Tamil
    DEU = "deu"  # German
    # Test languages
    ENG = "eng"  # English
    FRA = "fra"  # French
    CMN = "cmn"  # Mandarin Chinese
    JPN = "jpn"  # Japanese
    WOL = "wol"  # Wolof
    EUS = "eus"  # Basque

    @classmethod
    def from_commonvoice(cls, code: str) -> "Language":
        return {
            "de": cls.DEU,
            "en": cls.ENG,
            "fr": cls.FRA,
            "ja": cls.JPN,
            "sw": cls.SWA,
            "ta": cls.TAM,
            "th": cls.THA,
            "tr": cls.TUR,
            "uk": cls.UKR,
            "zh-CN": cls.CMN,
        }[code]

    @property
    def split(self) -> Literal["dev", "test"]:
        if self in {Language.UKR, Language.THA, Language.TUR, Language.SWA, Language.TAM, Language.DEU}:
            return "dev"
        return "test"
