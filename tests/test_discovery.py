"""End-to-end tests for the phoneme discovery evaluation pipeline."""

import pytest

from discophon.evaluate.discovery import phoneme_discovery
from discophon.validate import ValidateSameKeysError


def test_returns_all_metric_keys() -> None:
    units = {"f": [0, 1]}
    phones = {"f": ["x", "x", "y", "y"]}
    result = phoneme_discovery(units, phones, kind="many-to-one", n_units=2, n_phonemes=2)
    assert set(result) == {"pnmi", "per", "f1", "r_val"}


def test_perfect_units_give_ideal_scores() -> None:
    # units are in one-to-one correspondence with phones at the right rate
    units = {"f": [0, 1, 0, 1]}
    phones = {"f": ["x", "x", "y", "y", "x", "x", "y", "y"]}
    result = phoneme_discovery(units, phones, kind="many-to-one", n_units=2, n_phonemes=2)
    assert result["pnmi"] == 1.0
    assert result["per"] == 0.0
    assert result["f1"] == 1.0


def test_one_to_one_kind_runs() -> None:
    # one-to-one needs a square matrix: n_units == n_phonemes + 1 (the extra unit for silence)
    units = {"f": [0, 1, 2]}
    phones = {"f": ["a", "a", "b", "b", "c", "c"]}
    result = phoneme_discovery(units, phones, kind="one-to-one", n_units=4, n_phonemes=3)
    assert 0.0 <= result["pnmi"] <= 1.0
    assert result["per"] >= 0.0


def test_language_argument_infers_phoneme_count() -> None:
    # English has 39 phonemes; passing the language must be accepted in place of n_phonemes
    units = {"f": [0, 1]}
    phones = {"f": ["x", "x", "y", "y"]}
    result = phoneme_discovery(units, phones, kind="many-to-one", n_units=2, language="english")
    assert set(result) == {"pnmi", "per", "f1", "r_val"}


def test_does_not_raise_for_a_zero_precision_system() -> None:
    # Predicts a boundary every 20 ms, but the gold has a single boundary in the middle:
    # precision is ~0. This used to raise ZeroDivisionError via os/r_val.
    units = {"f": [0, 1, 0, 1, 0, 1, 0, 1]}
    phones = {"f": ["a"] * 8 + ["b"] * 8}
    result = phoneme_discovery(units, phones, kind="many-to-one", n_units=2, n_phonemes=2)
    assert isinstance(result["r_val"], float)
    assert isinstance(result["f1"], float)


def test_rejects_mismatched_keys() -> None:
    with pytest.raises(ValidateSameKeysError):
        phoneme_discovery({"a": [0]}, {"b": ["x"]}, kind="many-to-one", n_units=1, n_phonemes=1)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_metrics_are_finite_floats(seed: int) -> None:
    units = {"f": [(i + seed) % 3 for i in range(6)]}
    phones = {"f": ["a", "a", "b", "b", "c", "c", "a", "a", "b", "b", "c", "c"]}
    result = phoneme_discovery(units, phones, kind="many-to-one", n_units=3, n_phonemes=3)
    assert all(isinstance(v, float) for v in result.values())
    assert 0.0 <= result["pnmi"] <= 1.0 + 1e-6
    assert result["per"] >= 0.0
