from typing import Literal

# English, French, Mandarin Chinese, Japanese, Wolof, Basque
type TestLanguage = Literal["eng", "fra", "cmn", "jpn", "wol", "eus"]
# Ukrainian, Thai, Turkish, Swahili, Tamil, German
type DevLanguage = Literal["ukr", "tha", "tur", "swa", "tam", "deu"]
type Language = Literal[DevLanguage, TestLanguage]

COMMONVOICE_TO_ISO6393 = {
    "de": "deu",
    "en": "eng",
    "eu": "eus",
    "fr": "fra",
    "ja": "jpn",
    "sw": "swa",
    "ta": "tam",
    "th": "tha",
    "tr": "tur",
    "uk": "ukr",
    "zh-CN": "cmn",
}
