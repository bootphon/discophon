"""Tests for language resolution and metadata."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from discophon.languages import (
    ISO6393_TO_CV,
    Language,
    all_languages,
    commonvoice_languages,
    dev_languages,
    get_language,
    languages_in_split,
    load_phonemes,
)
from discophon.languages import test_languages as get_test_languages  # aliased: avoids pytest collecting it


def test_resolves_by_english_name() -> None:
    lang = get_language("German")
    assert lang.iso_639_3 == "deu"
    assert lang.name == "German"


def test_resolution_is_case_insensitive() -> None:
    assert get_language("GERMAN") == get_language("german") == get_language("German")


def test_resolves_by_iso_code() -> None:
    assert get_language("eng").name == "English"


def test_resolves_aliases() -> None:
    assert get_language("mandarin") == get_language("chinese") == get_language("cmn")


def test_resolves_common_voice_locale() -> None:
    assert get_language("zh-CN").iso_639_3 == "cmn"


def test_passthrough_of_language_instance() -> None:
    lang = get_language("french")
    assert get_language(lang) is lang


def test_unknown_language_raises() -> None:
    with pytest.raises(ValueError, match="Unknown language"):
        get_language("klingon")


@given(st.sampled_from(all_languages()))
def test_phoneme_count_matches_inventory(lang: Language) -> None:
    assert lang.n_phonemes == len(lang.phonemes)


def test_language_post_init_rejects_inconsistent_count() -> None:
    with pytest.raises(ValueError, match="Internal failure"):
        Language(name="German", iso_639_3="deu", split="dev", n_phonemes=999)


def test_split_partitions_languages() -> None:
    dev, test = dev_languages(), get_test_languages()
    assert len(dev) == 6
    assert len(test) == 6
    assert set(dev).isdisjoint(test)
    assert set(all_languages()) == set(dev) | set(test)


def test_all_languages_belong_to_their_declared_split() -> None:
    assert all(lang.split == "dev" for lang in dev_languages())
    assert all(lang.split == "test" for lang in get_test_languages())


def test_languages_in_split() -> None:
    assert languages_in_split("dev") == dev_languages()
    assert languages_in_split("test") == get_test_languages()


def test_languages_in_split_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown split"):
        languages_in_split("train")  # ty: ignore[invalid-argument-type]


def test_common_voice_languages_have_cv_locale_mapping() -> None:
    for lang in commonvoice_languages():
        assert lang.iso_639_3 in ISO6393_TO_CV
        # the CV locale must round-trip back to the same language
        assert get_language(ISO6393_TO_CV[lang.iso_639_3]) == lang


def test_iso_codes_are_unique() -> None:
    codes = [lang.iso_639_3 for lang in all_languages()]
    assert len(codes) == len(set(codes))


def test_all_languages_have_phoneme_inventories() -> None:
    inventory = load_phonemes()
    for lang in all_languages():
        assert lang.iso_639_3 in inventory
