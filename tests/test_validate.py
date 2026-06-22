"""Tests for the validation helpers."""

import pytest
from hypothesis import given
from hypothesis import strategies as st

from discophon.languages import get_language
from discophon.validate import (
    ArgumentsError,
    NumberPhonemesError,
    ValidateSameKeysError,
    infer_number_of_phonemes,
    validate_first_two_arguments_same_keys,
)


@validate_first_two_arguments_same_keys
def _compare(a: dict, b: dict) -> str:  # noqa: ARG001 -- the decorator only inspects the arguments' keys
    return "ok"


def test_accepts_dicts_with_same_keys() -> None:
    assert _compare({"x": 1, "y": 2}, {"x": 9, "y": 8}) == "ok"


def test_rejects_too_few_arguments() -> None:
    with pytest.raises(ArgumentsError):
        _compare({"x": 1})  # ty: ignore[missing-argument]


def test_rejects_different_keys() -> None:
    with pytest.raises(ValidateSameKeysError):
        _compare({"x": 1}, {"y": 2})


def test_rejects_non_dict_arguments() -> None:
    with pytest.raises(ValidateSameKeysError):
        _compare([1], [2])  # ty:ignore[invalid-argument-type]


@given(st.sets(st.text(min_size=1, max_size=5), max_size=6))
def test_same_keys_always_accepted(keys: set[str]) -> None:
    a = dict.fromkeys(keys, 1)
    b = dict.fromkeys(keys, 2)
    assert _compare(a, b) == "ok"


def test_infer_from_n_phonemes() -> None:
    assert infer_number_of_phonemes(39, None) == 39


def test_infer_from_language_name() -> None:
    assert infer_number_of_phonemes(None, "english") == 39


def test_infer_from_language_instance() -> None:
    lang = get_language("german")
    assert infer_number_of_phonemes(None, lang) == lang.n_phonemes


def test_infer_rejects_both_set() -> None:
    with pytest.raises(NumberPhonemesError):
        infer_number_of_phonemes(39, "english")


def test_infer_rejects_neither_set() -> None:
    with pytest.raises(NumberPhonemesError):
        infer_number_of_phonemes(None, None)
