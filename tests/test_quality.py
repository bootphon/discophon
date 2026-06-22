"""Tests for unit quality metrics: PNMI and P(phone|unit)."""

import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from xarray import DataArray

from discophon.evaluate.quality import pnmi, probability_phone_given_unit

TOL = 1e-6


def make_coocurrence(matrix: list[list[int]]) -> DataArray:
    arr = np.asarray(matrix, dtype=float)
    return DataArray(
        arr,
        dims=["phone", "unit"],
        coords=[[f"p{i}" for i in range(arr.shape[0])], list(range(arr.shape[1]))],
    )


def test_pnmi_is_one_for_perfect_correspondence() -> None:
    # diagonal: each phone maps to exactly one unit and vice versa
    cooc = make_coocurrence([[10, 0], [0, 10]])
    assert pnmi(cooc) == pytest.approx(1.0, abs=1e-6)


def test_pnmi_is_zero_for_independent_uniform() -> None:
    # every phone equally likely under every unit -> no mutual information
    cooc = make_coocurrence([[1, 1], [1, 1]])
    assert pnmi(cooc) == pytest.approx(0.0, abs=1e-6)


def test_pnmi_merging_units_keeps_perfect_score() -> None:
    # two units that both deterministically signal one phone still give PNMI 1
    cooc = make_coocurrence([[6, 4, 0], [0, 0, 10]])
    assert pnmi(cooc) == pytest.approx(1.0, abs=1e-6)


def test_pnmi_partial_overlap_is_between_zero_and_one() -> None:
    cooc = make_coocurrence([[8, 2], [3, 7]])
    value = pnmi(cooc)
    assert 0.0 < value < 1.0


@given(
    st.lists(
        st.lists(st.integers(min_value=0, max_value=50), min_size=2, max_size=5),
        min_size=2,
        max_size=5,
    )
)
def test_pnmi_is_bounded(rows: list[list[int]]) -> None:
    width = len(rows[0])
    matrix = [row[:width] + [0] * (width - len(row)) for row in rows]
    arr = np.asarray(matrix, dtype=float)
    # PNMI = MI / H(phone) is in [0, 1] only when the phone entropy is well defined,
    # i.e. at least two phones carry mass. The single-phone case has H(phone) ~= 0 and
    # is numerically unstable (it is never produced by the real pipeline).
    assume((arr.sum(axis=1) > 0).sum() >= 2)
    value = pnmi(make_coocurrence(matrix))
    assert -TOL <= value <= 1.0 + TOL


def test_probability_columns_sum_to_one() -> None:
    cooc = make_coocurrence([[2, 0, 1], [0, 4, 3]])
    proba = probability_phone_given_unit(cooc)
    assert np.allclose(proba.sum(dim="phone").values, 1.0)


def test_probability_drops_unused_units() -> None:
    # unit 1 never occurs -> it is removed from the conditional distribution
    cooc = make_coocurrence([[3, 0, 1], [2, 0, 5]])
    proba = probability_phone_given_unit(cooc)
    assert proba.sizes["unit"] == 2


def test_probability_is_named() -> None:
    proba = probability_phone_given_unit(make_coocurrence([[1, 2], [3, 4]]))
    assert proba.name == "P(phone|unit)"
