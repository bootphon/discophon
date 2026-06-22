"""Tests for unit quality metrics: PNMI and P(phone|unit)."""

import numpy as np
import pytest
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from xarray import DataArray

from discophon.evaluate.quality import pnmi, probability_phone_given_unit


def make_cooccurrence(matrix: list[list[int]]) -> DataArray:
    arr = np.asarray(matrix, dtype=float)
    return DataArray(
        arr,
        dims=["phone", "unit"],
        coords=[[f"p{i}" for i in range(arr.shape[0])], list(range(arr.shape[1]))],
    )


def test_pnmi_is_one_for_perfect_correspondence() -> None:
    # diagonal: each phone maps to exactly one unit and vice versa
    cooc = make_cooccurrence([[10, 0], [0, 10]])
    assert pnmi(cooc) == 1


def test_pnmi_is_zero_for_independent_uniform() -> None:
    # every phone equally likely under every unit -> mutual information is exactly 0
    cooc = make_cooccurrence([[1, 1], [1, 1]])
    assert pnmi(cooc) == 0.0


def test_pnmi_merging_units_keeps_perfect_score() -> None:
    # two units that both deterministically signal one phone still give PNMI 1
    cooc = make_cooccurrence([[6, 4, 0], [0, 0, 10]])
    assert pnmi(cooc) == 1


def test_pnmi_partial_overlap_is_between_zero_and_one() -> None:
    cooc = make_cooccurrence([[8, 2], [3, 7]])
    value = pnmi(cooc)
    assert 0.0 < value < 1.0


def _legacy_pnmi(cooccurrence: DataArray) -> float:
    """The original epsilon-based PNMI, kept as a reference oracle for the well-defined regime."""
    count = cooccurrence.values
    eps = 1e-10
    proba = count / count.sum()
    px, py = proba.sum(axis=1, keepdims=True), proba.sum(axis=0, keepdims=True)
    mutual_info = (proba * np.log(proba / (px @ py + eps) + eps)).sum()
    entropy_x = (-px * np.log(px + eps)).sum()
    return (mutual_info / entropy_x).item()


@settings(max_examples=1000)
@given(
    st.lists(
        st.lists(st.integers(min_value=0, max_value=512), min_size=1, max_size=50),
        min_size=1,
        max_size=50,
    )
)
def test_pnmi_matches_legacy_on_well_defined_inputs(rows: list[list[int]]) -> None:
    width = len(rows[0])
    matrix = [row[:width] + [0] * (width - len(row)) for row in rows]
    arr = np.asarray(matrix, dtype=float)
    assume((arr.sum(axis=1) > 0).sum() >= 2)
    cooc = make_cooccurrence(matrix)
    assert pnmi(cooc) == pytest.approx(_legacy_pnmi(cooc), abs=1e-6)


@given(
    st.lists(
        st.lists(st.integers(min_value=0, max_value=50), min_size=2, max_size=5),
        min_size=2,
        max_size=5,
    ),
    st.data(),
)
def test_pnmi_invariant_under_permutation(rows: list[list[int]], data: st.DataObject) -> None:
    # PNMI is a function of the joint distribution, so relabelling phones (rows) or units (columns)
    # must leave it unchanged. This pins down the metric without reusing the implementation's formula.
    width = len(rows[0])
    matrix = np.asarray([row[:width] + [0] * (width - len(row)) for row in rows], dtype=float)
    assume((matrix.sum(axis=1) > 0).sum() >= 2)
    base = pnmi(make_cooccurrence(matrix.tolist()))
    row_perm = data.draw(st.permutations(range(matrix.shape[0])))
    col_perm = data.draw(st.permutations(range(matrix.shape[1])))
    permuted = matrix[np.ix_(row_perm, col_perm)]
    assert pnmi(make_cooccurrence(permuted.tolist())) == pytest.approx(base, abs=1e-12)


@given(
    st.lists(
        st.tuples(st.integers(min_value=0, max_value=4), st.integers(min_value=1, max_value=50)),
        min_size=2,
        max_size=8,
    )
)
def test_pnmi_is_one_when_units_are_phone_deterministic(assignments: list[tuple[int, int]]) -> None:
    # Each unit puts all its mass on a single phone -> the unit determines the phone exactly,
    # so MI equals the phone entropy and PNMI is 1. Independent of the legacy oracle.
    assume(len({phone for phone, _ in assignments}) >= 2)  # need >= 2 phones with mass for H(phone) > 0
    n_phones = max(phone for phone, _ in assignments) + 1
    matrix = np.zeros((n_phones, len(assignments)), dtype=float)
    for unit, (phone, count) in enumerate(assignments):
        matrix[phone, unit] = count
    assert pnmi(make_cooccurrence(matrix.tolist())) == pytest.approx(1.0, abs=1e-9)


def test_pnmi_single_phone_is_defined_as_zero() -> None:
    # Only one phone carries mass -> phone entropy is 0. The legacy code returned ~1.0 (unstable);
    # the new implementation defines this degenerate case as 0.0.
    assert pnmi(make_cooccurrence([[5, 3], [0, 0]])) == 0.0


def test_pnmi_empty_matrix_is_defined_as_zero() -> None:
    assert pnmi(make_cooccurrence([[0, 0], [0, 0]])) == 0.0


def test_probability_columns_sum_to_one() -> None:
    cooc = make_cooccurrence([[2, 0, 1], [0, 4, 3]])
    proba = probability_phone_given_unit(cooc)
    assert np.allclose(proba.sum(dim="phone").values, 1.0)


def test_probability_drops_unused_units() -> None:
    # unit 1 never occurs -> it is removed from the conditional distribution
    cooc = make_cooccurrence([[3, 0, 1], [2, 0, 5]])
    proba = probability_phone_given_unit(cooc)
    assert proba.sizes["unit"] == 2


def test_probability_is_named() -> None:
    proba = probability_phone_given_unit(make_cooccurrence([[1, 2], [3, 4]]))
    assert proba.name == "P(phone|unit)"
