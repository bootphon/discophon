"""Tests for unit-to-phone alignment, the cooccurrence matrix, and assignments."""

import numpy as np
import pytest
from hypothesis import given
from xarray import DataArray

from discophon.evaluate.assignment import (
    NonSquareCooccurrenceError,
    align_units_and_phones,
    cooccurrence_matrix,
    mapping_many_to_one,
    mapping_one_to_one,
    phone_assignments,
    relabel_assignment,
)
from discophon.validate import ValidateSameKeysError

from .strategies import int_sequences


def test_align_repeats_units_to_match_phone_rate() -> None:
    # step_units=20, step_phones=10 -> each unit is repeated twice
    aligned = align_units_and_phones({"f": [5, 6]}, {"f": ["a", "a", "b", "b"]}, step_units=20, step_phones=10)
    assert aligned["f"]["units"] == [5, 5, 6, 6]
    assert aligned["f"]["phones"] == ["a", "a", "b", "b"]


def test_align_truncates_to_shortest_within_margin() -> None:
    # one extra phone (<= repeat) is tolerated and trimmed
    aligned = align_units_and_phones({"f": [5, 6]}, {"f": ["a", "a", "b", "b", "b"]}, step_units=20, step_phones=10)
    assert len(aligned["f"]["units"]) == len(aligned["f"]["phones"]) == 4


def test_align_raises_when_lengths_differ_too_much() -> None:
    with pytest.raises(ValueError, match="tokens of differences"):
        align_units_and_phones({"f": [5]}, {"f": ["a", "a", "a", "a", "a"]}, step_units=20, step_phones=10)


@given(int_sequences(min_size=1, max_size=20))
def test_align_lengths_are_equal_per_file(units: list[int]) -> None:
    phones = {"f": ["a"] * (2 * len(units))}
    aligned = align_units_and_phones({"f": units}, phones, step_units=20, step_phones=10)
    assert len(aligned["f"]["units"]) == len(aligned["f"]["phones"])


def test_cooccurrence_shape_and_counts() -> None:
    cooc = cooccurrence_matrix({"f": [0, 1]}, {"f": ["x", "x", "y", "y"]}, n_units=2, n_phonemes=2)
    assert cooc.shape == (3, 2)  # n_phonemes + 1 (SIL slot), n_units
    assert cooc.dims == ("phone", "unit")
    # unit 0 always under phone x, unit 1 always under phone y
    assert cooc.sel(phone="x", unit=0).item() == 2
    assert cooc.sel(phone="y", unit=1).item() == 2
    assert cooc.sel(phone="x", unit=1).item() == 0


def test_cooccurrence_total_equals_aligned_tokens() -> None:
    cooc = cooccurrence_matrix({"f": [0, 1, 2]}, {"f": ["a", "a", "b", "b", "c", "c"]}, n_units=3, n_phonemes=3)
    assert int(cooc.sum().item()) == 6


def test_cooccurrence_rows_sorted_by_frequency_descending() -> None:
    # phone "a" appears far more often than "b"
    units = {"f": [0] * 5 + [1]}
    phones = {"f": ["a"] * 10 + ["b"] * 2}
    cooc = cooccurrence_matrix(units, phones, n_units=2, n_phonemes=2)
    row_sums = cooc.sum(dim="unit").values
    assert list(row_sums) == sorted(row_sums, reverse=True)


def test_cooccurrence_pads_missing_phonemes() -> None:
    # only one phone observed but language has 2 -> a "<missing>" row is added plus the SIL slot
    cooc = cooccurrence_matrix({"f": [0]}, {"f": ["a", "a"]}, n_units=1, n_phonemes=2)
    assert cooc.shape == (3, 1)
    assert any("missing" in str(p) for p in cooc["phone"].values)


def test_cooccurrence_raises_when_unit_out_of_range() -> None:
    with pytest.raises(IndexError):
        cooccurrence_matrix({"f": [0, 5]}, {"f": ["a", "a", "b", "b"]}, n_units=2, n_phonemes=2)


def test_cooccurrence_raises_when_too_many_phonemes() -> None:
    with pytest.raises(IndexError):
        cooccurrence_matrix({"f": [0, 0, 0]}, {"f": ["a", "a", "b", "b", "c", "c"]}, n_units=1, n_phonemes=1)


def test_cooccurrence_empty_input_is_all_zeros() -> None:
    # empty sequences must not crash (np.array([]) would default to float and break np.bincount)
    cooc = cooccurrence_matrix({"f": []}, {"f": []}, n_units=2, n_phonemes=2)
    assert cooc.shape == (3, 2)
    assert int(cooc.sum().item()) == 0


def test_cooccurrence_rejects_mismatched_keys() -> None:
    with pytest.raises(ValidateSameKeysError):
        cooccurrence_matrix({"a": [0]}, {"b": ["x"]}, n_units=1, n_phonemes=1)


def test_mapping_many_to_one_picks_argmax_phone() -> None:
    cooc = DataArray(
        np.array([[5, 1], [2, 9]]),
        dims=["phone", "unit"],
        coords=[["a", "b"], [0, 1]],
    )
    assert mapping_many_to_one(cooc) == {0: "a", 1: "b"}


def test_mapping_many_to_one_covers_every_unit() -> None:
    cooc = DataArray(
        np.array([[3, 1, 4], [1, 5, 2]]),
        dims=["phone", "unit"],
        coords=[["a", "b"], [0, 1, 2]],
    )
    mapping = mapping_many_to_one(cooc)
    assert set(mapping) == {0, 1, 2}  # all units mapped, possibly many-to-one


def test_mapping_one_to_one_is_a_bijection() -> None:
    cooc = DataArray(
        np.array([[10, 1], [1, 10]]),
        dims=["phone", "unit"],
        coords=[["a", "b"], [0, 1]],
    )
    mapping = mapping_one_to_one(cooc)
    assert mapping == {0: "a", 1: "b"}
    # bijection: distinct units map to distinct phones
    assert len(set(mapping.values())) == len(mapping)


def test_mapping_one_to_one_rejects_non_square_matrix() -> None:
    # 2 phones x 3 units: a one-to-one mapping would leave a unit unassigned -> KeyError downstream
    cooc = DataArray(
        np.array([[3, 1, 4], [1, 5, 2]]),
        dims=["phone", "unit"],
        coords=[["a", "b"], [0, 1, 2]],
    )
    with pytest.raises(NonSquareCooccurrenceError):
        mapping_one_to_one(cooc)


def test_phone_assignments_one_to_one_rejects_non_square_matrix() -> None:
    # n_units (5) > n_phonemes + 1 (4) -> non-square cooccurrence
    units = {"f": [0, 1, 2, 3, 4]}
    phones = {"f": ["a", "a", "b", "b", "c", "c", "a", "a", "b", "b"]}
    cooc = cooccurrence_matrix(units, phones, n_units=5, n_phonemes=3)
    with pytest.raises(NonSquareCooccurrenceError):
        phone_assignments(units, cooc, kind="one-to-one")


def test_mapping_one_to_one_maximizes_total_weight() -> None:
    # crossed assignment is optimal here
    cooc = DataArray(
        np.array([[1, 8], [9, 2]]),
        dims=["phone", "unit"],
        coords=[["a", "b"], [0, 1]],
    )
    assert mapping_one_to_one(cooc) == {1: "a", 0: "b"}


def test_phone_assignments_applies_mapping() -> None:
    cooc = cooccurrence_matrix({"f": [0, 1]}, {"f": ["x", "x", "y", "y"]}, n_units=2, n_phonemes=2)
    out = phone_assignments({"f": [0, 0, 1]}, cooc, kind="many-to-one")
    assert out == {"f": ["x", "x", "y"]}


def test_phone_assignments_unknown_kind_raises() -> None:
    cooc = cooccurrence_matrix({"f": [0, 1]}, {"f": ["x", "x", "y", "y"]}, n_units=2, n_phonemes=2)
    with pytest.raises(ValueError, match="Unknown kind"):
        phone_assignments({"f": [0]}, cooc, kind="bogus")  # ty:ignore[invalid-argument-type]


def _proba(matrix: list[list[float]]) -> DataArray:
    arr = np.asarray(matrix, dtype=float)
    return DataArray(
        arr,
        dims=["phone", "unit"],
        coords=[[f"p{i}" for i in range(arr.shape[0])], list(range(arr.shape[1]))],
        name="P(phone|unit)",
    )


def test_relabel_preserves_unit_axis_and_name() -> None:
    proba = _proba([[0.9, 0.1, 0.2, 0.8], [0.1, 0.9, 0.8, 0.2]])
    out = relabel_assignment([0, 1, 1, 0], proba)
    assert out.dims == ("unit",)
    assert out.name == "assignment"
    assert out["unit"].values.tolist() == [0, 1, 2, 3]


def test_relabel_preserves_the_partition() -> None:
    # units sharing a cluster before must still share one after relabeling
    proba = _proba([[0.9, 0.1, 0.2, 0.8], [0.1, 0.9, 0.8, 0.2]])
    original = [0, 1, 1, 0]
    relabeled = relabel_assignment(original, proba).values.tolist()
    pairs = list(zip(original, relabeled, strict=True))
    assert len({new for old, new in pairs if old == 0}) == 1
    assert len({new for old, new in pairs if old == 1}) == 1
    assert len(set(relabeled)) == len(set(original))


def test_relabel_orders_clusters_by_most_probable_phone() -> None:
    # cluster 0's most probable phone is p1 (index 1); cluster 1's is p0 (index 0) -> labels swap
    proba = _proba([[0.1, 0.9], [0.9, 0.1]])
    out = relabel_assignment([0, 1], proba).values.tolist()
    assert out == [1, 0]
