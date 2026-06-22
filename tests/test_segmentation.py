"""Tests for phone segmentation: metrics algebra, boundaries, and boundary comparison."""

from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from discophon.evaluate.segmentation import (
    Boundaries,
    SegmentationEvaluation,
    compare_boundaries,
    phone_segmentation,
)
from discophon.validate import ValidateSameKeysError

counts = st.integers(min_value=0, max_value=1000)


def test_perfect_scores() -> None:
    seg = SegmentationEvaluation(true_positives=10, false_positives=0, false_negatives=0)
    assert seg.precision == pytest.approx(1.0)
    assert seg.recall == pytest.approx(1.0)
    assert seg.f1 == pytest.approx(1.0)
    assert seg.os == pytest.approx(0.0)
    assert seg.r_val == pytest.approx(1.0)


@given(tp=st.integers(1, 1000), fp=counts, fn=counts)
def test_f1_is_harmonic_mean(tp: int, fp: int, fn: int) -> None:
    seg = SegmentationEvaluation(tp, fp, fn)
    expected = 2 * seg.precision * seg.recall / (seg.precision + seg.recall)
    assert seg.f1 == pytest.approx(expected)


@given(tp=st.integers(1, 1000), fp=counts, fn=counts)
def test_metrics_are_bounded(tp: int, fp: int, fn: int) -> None:
    seg = SegmentationEvaluation(tp, fp, fn)
    assert 0.0 <= seg.precision <= 1.0
    assert 0.0 <= seg.recall <= 1.0
    assert 0.0 <= seg.f1 <= 1.0
    assert seg.r_val <= 1.0


@given(tp=st.integers(1, 1000), fp=counts, fn=counts)
def test_over_segmentation_formula(tp: int, fp: int, fn: int) -> None:
    seg = SegmentationEvaluation(tp, fp, fn)
    assert seg.os == pytest.approx(seg.recall / seg.precision - 1)


def test_over_segmentation_defined_when_precision_is_zero() -> None:
    # A system that predicts boundaries but gets none right has precision 0; os must not divide by it.
    seg = SegmentationEvaluation(true_positives=0, false_positives=3, false_negatives=2)
    assert seg.precision == pytest.approx(0.0)
    assert seg.os == pytest.approx((0 + 3) / (0 + 2) - 1)
    assert isinstance(seg.r_val, float)


@given(a=st.tuples(counts, counts, counts), b=st.tuples(counts, counts, counts))
def test_add_sums_counts(a: tuple[int, int, int], b: tuple[int, int, int]) -> None:
    total = SegmentationEvaluation(*a) + SegmentationEvaluation(*b)
    assert total.true_positives == a[0] + b[0]
    assert total.false_positives == a[1] + b[1]
    assert total.false_negatives == a[2] + b[2]


def test_add_identity_element() -> None:
    seg = SegmentationEvaluation(3, 1, 2)
    assert seg + SegmentationEvaluation(0, 0, 0) == seg


def test_add_rejects_other_types() -> None:
    with pytest.raises(NotImplementedError):
        _ = SegmentationEvaluation(1, 1, 1) + 5


def test_describe_contains_all_metrics() -> None:
    text = SegmentationEvaluation(10, 2, 3).describe()
    for token in ("Precision", "Recall", "F1", "OS", "R-val"):
        assert token in text


def test_boundaries_sorted_and_immutable() -> None:
    b = Boundaries([30, 10, 20])
    assert list(b.times) == [10, 20, 30]
    with pytest.raises(ValueError, match="read-only"):
        b.times[0] = 999


def test_boundaries_len() -> None:
    assert len(Boundaries([10, 20, 30])) == 3
    assert len(Boundaries([])) == 0


def test_boundaries_str_renders_seconds() -> None:
    # times are stored in ms and printed in seconds
    text = str(Boundaries([1000, 2000]))
    assert "1.0s" in text
    assert "2.0s" in text


def test_from_tokens_marks_group_transitions() -> None:
    # groups [a,a],[b,b,b],[c] -> cumulative 2,5,6 times *10 = 20,50,60, drop the last
    b = Boundaries.from_tokens(["a", "a", "b", "b", "b", "c"], 10)
    assert list(b.times) == [20, 50]


@given(st.lists(st.sampled_from("abc"), min_size=1, max_size=40), st.integers(1, 50))
def test_from_tokens_count_equals_groups_minus_one(tokens: list[str], step: int) -> None:
    n_groups = 1 + sum(1 for a, b in pairwise(tokens) if a != b)
    assert len(Boundaries.from_tokens(tokens, step)) == n_groups - 1


@given(st.lists(st.integers(0, 5000), max_size=30), st.integers(1, 100))
def test_tolerance_windows_are_ordered_clipped_and_non_overlapping(times: list[int], margin: int) -> None:
    windows = Boundaries(times).tolerance(margin)
    assert (windows >= 0).all()  # clipped at zero
    assert (windows[:, 1] >= windows[:, 0]).all()  # each window is well-formed
    assert (windows[:-1, 1] <= windows[1:, 0]).all()  # consecutive windows never overlap


def test_compare_identical_is_all_true_positives() -> None:
    b = Boundaries([100, 200, 300])
    result = compare_boundaries(b, b, margin_in_ms=20)
    assert result == SegmentationEvaluation(true_positives=3, false_positives=0, false_negatives=0)


def test_compare_prediction_within_margin_counts() -> None:
    result = compare_boundaries(Boundaries([100]), Boundaries([115]), margin_in_ms=20)
    assert result.true_positives == 1
    assert result.false_positives == 0


def test_compare_prediction_outside_margin_misses() -> None:
    result = compare_boundaries(Boundaries([100]), Boundaries([200]), margin_in_ms=20)
    assert result.true_positives == 0
    assert result.false_negatives == 1
    assert result.false_positives == 1


def test_compare_no_predictions() -> None:
    result = compare_boundaries(Boundaries([10, 20]), Boundaries([]), margin_in_ms=20)
    assert result == SegmentationEvaluation(true_positives=0, false_positives=0, false_negatives=2)


def test_phone_segmentation_identical_sequences_score_perfectly() -> None:
    phones = {"f1": ["a", "a", "b", "c"], "f2": ["x", "y", "y"]}
    # step_units == step_phones so the boundaries line up exactly
    result = phone_segmentation(phones, phones, step_units=10, step_phones=10, margin_in_ms=20)
    assert result.false_positives == 0
    assert result.false_negatives == 0
    assert result.f1 == pytest.approx(1.0)


def test_phone_segmentation_aggregates_over_files() -> None:
    gold = {"f1": ["a", "b"], "f2": ["a", "b"]}
    pred = {"f1": ["a", "b"], "f2": ["a", "b"]}
    result = phone_segmentation(pred, gold, step_units=10, step_phones=10)
    assert result.true_positives == 2  # one boundary per file, both detected


def test_phone_segmentation_rejects_mismatched_keys() -> None:
    with pytest.raises(ValidateSameKeysError):
        phone_segmentation({"a": ["x", "y"]}, {"b": ["x", "y"]})
