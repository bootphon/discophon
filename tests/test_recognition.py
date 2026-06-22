"""Tests for phone recognition: deduplication, edit distance, and phone error rate."""

from itertools import pairwise

import numpy as np
import pytest
from hypothesis import given

from discophon.evaluate.recognition import deduplicate, edit_distance, phone_error_rate
from discophon.validate import ArgumentsError, ValidateSameKeysError

from .strategies import int_sequences, phone_sequences, reference_edit_distance


def test_deduplicate_collapses_consecutive_runs() -> None:
    assert deduplicate([1, 1, 2, 2, 2, 3, 1, 1]) == [1, 2, 3, 1]


def test_deduplicate_empty_raises() -> None:
    with pytest.raises(ValueError, match="Empty sequence"):
        deduplicate([])


@given(phone_sequences(min_size=1))
def test_deduplicate_no_consecutive_duplicates_remain(seq: list[str]) -> None:
    out = deduplicate(seq)
    assert all(a != b for a, b in pairwise(out))


@given(phone_sequences(min_size=1))
def test_deduplicate_idempotent(seq: list[str]) -> None:
    once = deduplicate(seq)
    assert deduplicate(once) == once


@given(phone_sequences(min_size=1))
def test_deduplicate_never_longer_and_preserves_set(seq: list[str]) -> None:
    out = deduplicate(seq)
    assert len(out) <= len(seq)
    assert set(out) == set(seq)


def test_edit_distance_known_values() -> None:
    assert edit_distance(np.array([1, 2, 3]), np.array([1, 2, 4])) == 1
    assert edit_distance(np.array([1, 2, 3]), np.array([1, 2, 3])) == 0


def test_edit_distance_empty_sequences() -> None:
    assert edit_distance(np.array([], dtype=np.int64), np.array([1, 2], dtype=np.int64)) == 2
    assert edit_distance(np.array([1, 2], dtype=np.int64), np.array([], dtype=np.int64)) == 2
    assert edit_distance(np.array([], dtype=np.int64), np.array([], dtype=np.int64)) == 0


@given(int_sequences(), int_sequences())
def test_edit_distance_matches_reference(a: list[int], b: list[int]) -> None:
    assert edit_distance(np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)) == reference_edit_distance(a, b)


@given(int_sequences())
def test_edit_distance_identity(a: list[int]) -> None:
    assert edit_distance(np.array(a, dtype=np.int64), np.array(a, dtype=np.int64)) == 0


@given(int_sequences(), int_sequences())
def test_edit_distance_symmetry(a: list[int], b: list[int]) -> None:
    xa, xb = np.array(a, dtype=np.int64), np.array(b, dtype=np.int64)
    assert edit_distance(xa, xb) == edit_distance(xb, xa)


@given(int_sequences(), int_sequences())
def test_edit_distance_bounds(a: list[int], b: list[int]) -> None:
    d = edit_distance(np.array(a, dtype=np.int64), np.array(b, dtype=np.int64))
    assert abs(len(a) - len(b)) <= d <= max(len(a), len(b))


@given(int_sequences(max_size=12), int_sequences(max_size=12), int_sequences(max_size=12))
def test_edit_distance_triangle_inequality(a: list[int], b: list[int], c: list[int]) -> None:
    xa, xb, xc = (np.array(x, dtype=np.int64) for x in (a, b, c))
    assert edit_distance(xa, xc) <= edit_distance(xa, xb) + edit_distance(xb, xc)


def test_per_perfect_match_is_zero() -> None:
    gold = {"f1": ["a", "b", "c"], "f2": ["a", "a", "b"]}
    assert phone_error_rate(gold, gold, n_jobs=1) == pytest.approx(0.0)


def test_per_deduplicates_before_comparing() -> None:
    # Repeated tokens collapse, so this is a perfect match after dedup.
    assert phone_error_rate({"f": ["a", "a", "b"]}, {"f": ["a", "b"]}, n_jobs=1) == pytest.approx(0.0)


def test_per_value_matches_manual_computation() -> None:
    # dedup(pred)=[a,c], dedup(gold)=[a,b]; edit distance 1, gold length 2 -> 0.5
    assert phone_error_rate({"f": ["a", "a", "c"]}, {"f": ["a", "b"]}, n_jobs=1) == pytest.approx(0.5)


def test_per_aggregates_across_files() -> None:
    pred = {"f1": ["a", "x"], "f2": ["b", "b"]}
    gold = {"f1": ["a", "b"], "f2": ["b", "b"]}
    # total edits = 1 (f1) + 0 (f2); total gold length after dedup = 2 + 1 = 3
    assert phone_error_rate(pred, gold, n_jobs=1) == pytest.approx(1 / 3)


@given(phone_sequences(min_size=1), phone_sequences(min_size=1))
def test_per_is_non_negative(a: list[str], b: list[str]) -> None:
    assert phone_error_rate({"f": a}, {"f": b}, n_jobs=1) >= 0.0


def test_per_rejects_mismatched_keys() -> None:
    with pytest.raises(ValidateSameKeysError):
        phone_error_rate({"a": ["x"]}, {"b": ["x"]}, n_jobs=1)


def test_per_rejects_too_few_arguments() -> None:
    with pytest.raises(ArgumentsError):
        phone_error_rate({"a": ["x"]})  # ty: ignore[missing-argument]


def test_per_empty_gold_raises() -> None:
    # Empty annotations cannot be normalized; deduplicate rejects them before any division by zero.
    with pytest.raises(ValueError, match="Empty sequence"):
        phone_error_rate({"f": []}, {"f": []}, n_jobs=1)
