"""Tests for the validation helpers."""

from itertools import product, starmap
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st

from discophon.data import alignment_filename, item_filename, manifest_filename
from discophon.languages import all_languages, get_language
from discophon.validate import (
    ArgumentsError,
    DatasetError,
    NumberPhonemesError,
    ValidateSameKeysError,
    infer_number_of_phonemes,
    validate_dataset_structure,
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


def build_valid_dataset(root: Path) -> Path:
    """Create the minimal (empty-file) tree that ``validate_dataset_structure`` accepts."""
    languages = all_languages()
    for name in starmap(alignment_filename, product(languages, ["dev", "test"])):
        (root / "alignment" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "alignment" / name).touch()
    for kind, lang, split in product(["triphone", "phoneme"], languages, ["dev", "test"]):
        (root / "item").mkdir(parents=True, exist_ok=True)
        (root / "item" / item_filename(lang, split, kind=kind)).touch()
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    splits = ["dev", "test", "train-10h", "train-10min", "train-1h"]
    for name in starmap(manifest_filename, product(languages, splits)):
        (root / "manifest" / name).touch()
    (root / "manifest" / "speakers.jsonl").touch()
    for lang in languages:
        audio = root / "audio" / lang.iso_639_3
        audio.mkdir(parents=True, exist_ok=True)
        for split in ["all", "dev", "test", "train-10h", "train-10min", "train-1h"]:
            (audio / f"{split}.wav").touch()
    return root


def test_validate_dataset_structure_accepts_valid_tree(tmp_path: Path) -> None:
    validate_dataset_structure(build_valid_dataset(tmp_path))


def test_validate_dataset_structure_rejects_missing_top_level_dir(tmp_path: Path) -> None:
    root = build_valid_dataset(tmp_path)
    next((root / "alignment").glob("*")).parent.rename(root / "alignment_renamed")
    with pytest.raises(DatasetError):
        validate_dataset_structure(root)


def test_validate_dataset_structure_rejects_unexpected_extra_file(tmp_path: Path) -> None:
    root = build_valid_dataset(tmp_path)
    (root / "manifest" / "unexpected.csv").touch()
    with pytest.raises(DatasetError):
        validate_dataset_structure(root)


def test_validate_dataset_structure_rejects_missing_audio_split(tmp_path: Path) -> None:
    root = build_valid_dataset(tmp_path)
    next((root / "audio").glob("*/all.wav")).unlink()
    with pytest.raises(DatasetError):
        validate_dataset_structure(root)
