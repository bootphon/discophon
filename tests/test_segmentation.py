"""Tests for phone segmentation: metrics algebra, boundaries, and boundary comparison."""

import math
from itertools import pairwise

import pytest
from hypothesis import given
from hypothesis import strategies as st

from discophon.evaluate.segmentation import (
    Boundaries,
    NoBoundariesError,
    SegmentationEvaluation,
    compare_boundaries,
    phone_segmentation,
)
from discophon.validate import ValidateSameKeysError


def test_metrics_raise_clear_error_without_boundaries() -> None:
    # no gold and no predicted boundaries: metrics are undefined and must fail loudly, not with ZeroDivisionError
    empty = SegmentationEvaluation(true_positives=0, false_positives=0, false_negatives=0)
    for accessor in ("recall", "precision", "f1", "os", "r_val"):
        with pytest.raises(NoBoundariesError):
            getattr(empty, accessor)


counts = st.integers(min_value=0, max_value=1000)


def test_perfect_scores() -> None:
    seg = SegmentationEvaluation(true_positives=10, false_positives=0, false_negatives=0)
    assert seg.precision == 1.0
    assert seg.recall == 1.0
    assert seg.f1 == 1.0
    assert seg.os == 0.0
    assert seg.r_val == 1.0


@given(tp=st.integers(1, 1000), fp=counts, fn=counts)
def test_f1_is_harmonic_mean(tp: int, fp: int, fn: int) -> None:
    seg = SegmentationEvaluation(tp, fp, fn)
    expected = 2 * seg.precision * seg.recall / (seg.precision + seg.recall)
    assert seg.f1 == pytest.approx(expected, abs=1e-12)


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
    assert seg.os == pytest.approx(seg.recall / seg.precision - 1, abs=1e-12)


def test_over_segmentation_defined_when_precision_is_zero() -> None:
    # A system that predicts boundaries but gets none right has precision 0; os must not divide by it.
    seg = SegmentationEvaluation(true_positives=0, false_positives=3, false_negatives=2)
    assert seg.precision == 0.0
    assert seg.os == (0 + 3) / (0 + 2) - 1
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
def test_tolerance_windows_are_ordered_clipped_and_disjoint(times: list[int], margin: int) -> None:
    windows = Boundaries(times).tolerance(margin)
    assert (windows >= 0).all()  # clipped at zero
    assert (windows[:, 1] >= windows[:, 0]).all()  # each window is well-formed
    # strictly disjoint: a single prediction can never fall in two windows, so it cannot be double counted
    assert (windows[:-1, 1] < windows[1:, 0]).all()


@given(st.lists(st.integers(0, 5000), max_size=30), st.integers(1, 100))
def test_tolerance_window_contains_its_own_boundary(times: list[int], margin: int) -> None:
    # Splitting overlaps must never push a boundary out of its own window (this underpins perfect self-scoring).
    b = Boundaries(times)
    windows = b.tolerance(margin)
    assert (windows[:, 0] <= b.times).all()
    assert (b.times <= windows[:, 1]).all()


def test_boundaries_deduplicates_coincident_times() -> None:
    # A boundary is a point in time; coincident boundaries collapse to one (and stay sorted).
    assert list(Boundaries([30, 10, 10, 20, 30]).times) == [10, 20, 30]
    assert len(Boundaries([5, 5, 5])) == 1


def test_tolerance_reference_windows() -> None:
    # Far-apart boundaries keep the full +/-margin window; close ones are split at the boundary midpoint
    # (the earlier window keeps the midpoint, the later one starts just after) so the regions are disjoint.
    assert Boundaries([100]).tolerance(20).tolist() == [[80, 120]]
    assert Boundaries([100, 200]).tolerance(20).tolist() == [[80, 120], [180, 220]]
    assert Boundaries([100, 120]).tolerance(20).tolist() == [[80, 110], [111, 140]]  # overlapping
    assert Boundaries([100, 140]).tolerance(20).tolist() == [[80, 120], [121, 160]]  # exactly touching


def test_tolerance_clips_lower_edge_at_zero() -> None:
    assert Boundaries([5]).tolerance(20).tolist() == [[0, 25]]


def test_tolerance_splits_at_boundary_midpoint_near_zero() -> None:
    # Regression: the split must use the midpoint between boundary *times*, not between the clipped window
    # edges. Near t=0 the lower edges all clip to 0; deriving the split from them collapses the midpoints and
    # produces inverted windows. Boundaries 0, 1, 2 must give the disjoint, well-formed windows below.
    assert Boundaries([0, 1, 2]).tolerance(2).tolist() == [[0, 0], [1, 1], [2, 4]]


def test_tolerance_empty_boundaries_returns_no_windows() -> None:
    assert Boundaries([]).tolerance(20).shape == (0, 2)


def test_from_tokens_without_transitions_has_no_boundary() -> None:
    # A single group (or no tokens) has no internal transition, hence no boundary.
    assert len(Boundaries.from_tokens(["a", "a", "a"], 10)) == 0
    assert len(Boundaries.from_tokens([], 10)) == 0


@given(st.lists(st.integers(0, 5000), max_size=20), st.integers(0, 200))
def test_compare_self_is_all_true_positives(times: list[int], margin: int) -> None:
    # Comparing a segmentation against itself must score perfectly for any margin.
    b = Boundaries(times)
    assert compare_boundaries(b, b, margin_in_ms=margin) == SegmentationEvaluation(len(b), 0, 0)


@given(
    gold=st.lists(st.integers(0, 5000), max_size=20),
    pred=st.lists(st.integers(0, 5000), max_size=20),
    margins=st.tuples(st.integers(0, 200), st.integers(0, 200)),
)
def test_larger_margin_never_reduces_true_positives(
    gold: list[int], pred: list[int], margins: tuple[int, int]
) -> None:
    # A more tolerant margin only ever widens the detection windows, so it cannot lose a hit.
    small, large = sorted(margins)
    g, p = Boundaries(gold), Boundaries(pred)
    assert (
        compare_boundaries(g, p, margin_in_ms=large).true_positives
        >= compare_boundaries(g, p, margin_in_ms=small).true_positives
    )


@given(
    gold_times=st.lists(st.integers(0, 5000), max_size=20),
    pred_times=st.lists(st.integers(0, 5000), max_size=20),
    margin=st.integers(1, 100),
)
def test_compare_counts_stay_consistent(gold_times: list[int], pred_times: list[int], margin: int) -> None:
    # false_positives and false_negatives are derived by subtracting tp, so the only way they stay
    # non-negative is if each prediction is matched at most once: tp <= min(#gold, #pred).
    result = compare_boundaries(Boundaries(gold_times), Boundaries(pred_times), margin_in_ms=margin)
    assert 0 <= result.true_positives <= min(len(gold_times), len(pred_times))
    assert result.false_positives >= 0
    assert result.false_negatives >= 0


def test_compare_shared_midpoint_is_not_double_counted() -> None:
    # Gold boundaries 20 ms apart: their +/-20 ms windows overlap and are split at the midpoint 110.
    # A single prediction at 110 must count as one hit, not one for each gold boundary.
    result = compare_boundaries(Boundaries([100, 120]), Boundaries([110]), margin_in_ms=20)
    assert result.true_positives == 1
    assert result.false_positives == 0  # not -1
    assert result.false_negatives == 1


def test_compare_touching_windows_are_not_double_counted() -> None:
    # Gold boundaries exactly 2*margin apart: windows touch at 120. The shared edge must belong to one side.
    result = compare_boundaries(Boundaries([100, 140]), Boundaries([120]), margin_in_ms=20)
    assert result.true_positives == 1
    assert result.false_positives == 0


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
    assert result.f1 == 1.0


def test_phone_segmentation_aggregates_over_files() -> None:
    gold = {"f1": ["a", "b"], "f2": ["a", "b"]}
    pred = {"f1": ["a", "b"], "f2": ["a", "b"]}
    result = phone_segmentation(pred, gold, step_units=10, step_phones=10)
    assert result.true_positives == 2  # one boundary per file, both detected


def test_phone_segmentation_rejects_mismatched_keys() -> None:
    with pytest.raises(ValidateSameKeysError):
        phone_segmentation({"a": ["x", "y"]}, {"b": ["x", "y"]})


# --- Pathological cases and reference values from (Rasanen et al., 2009) ---------------------------------
# The "search region method" (sec. 2.3): a +/-margin region is placed around every gold boundary; overlapping
# regions are shrunk to a common midpoint so they stay disjoint. A region containing at least one predicted
# boundary is a single hit; any further predictions in it are insertions; empty regions are deletions.


def test_paper_figure1_single_boundary_in_overlap_counts_once() -> None:
    # Fig. 1: two gold boundaries within 2*margin, the algorithm produces a single boundary in their
    # overlapping search regions. Re-using it as a hit for both is the bug the paper rules out: it must
    # be a hit for the nearest boundary only, leaving the other as a deletion.
    result = compare_boundaries(Boundaries([100, 120]), Boundaries([105]), margin_in_ms=20)
    assert result == SegmentationEvaluation(true_positives=1, false_positives=0, false_negatives=1)


def test_paper_extra_predictions_in_one_region_are_insertions() -> None:
    # Sec. 2.3: "a region containing a boundary is a hit and all additional boundaries are insertions".
    # Three predictions all land within margin of a single gold boundary -> one hit, two insertions.
    result = compare_boundaries(Boundaries([100]), Boundaries([90, 100, 110]), margin_in_ms=20)
    assert result == SegmentationEvaluation(true_positives=1, false_positives=2, false_negatives=0)
    assert result.recall == 1.0
    assert result.precision == pytest.approx(1 / 3)


def test_paper_empty_region_is_a_deletion() -> None:
    # Sec. 2.3: "empty regions are considered as deletions". The far gold boundary gets no prediction.
    result = compare_boundaries(Boundaries([100, 500]), Boundaries([100]), margin_in_ms=20)
    assert result == SegmentationEvaluation(true_positives=1, false_positives=0, false_negatives=1)


def test_paper_stochastic_over_segmentation_inflates_recall_but_sinks_rvalue() -> None:
    # Sec. 3-4: an over-segmenting system (a boundary at every 20 ms frame) trivially covers the timeline,
    # so it detects every gold boundary (recall == 1) yet floods the output with insertions. The F-value
    # stays deceptively high while the R-value collapses below zero -- the paper's central argument.
    gold = {"f": ["a", "a", "b", "b", "c", "c"]}  # boundaries at 20, 40 ms (10 ms step)
    over_segmenting = {"f": ["x", "y", "x", "y", "x", "y"]}  # boundaries at 20, 40, 60, 80, 100 ms (20 ms step)
    result = phone_segmentation(over_segmenting, gold)
    assert result == SegmentationEvaluation(true_positives=2, false_positives=3, false_negatives=0)
    assert result.recall == 1.0
    assert result.os == pytest.approx(1.5)  # 150% over-segmentation
    assert result.f1 == pytest.approx(4 / 7)  # F-value still ~0.57
    assert result.r_val < 0.0  # but the R-value is negative
    assert result.r_val < result.f1  # and far below the F-value


# --- R-value reference points derived directly from the formula (eqs. 6-8) ------------------------------


def test_r_value_target_point_is_one() -> None:
    # Sec. 4: the "target point" of 100% hit-rate and 0% over-segmentation is the ideal operating point R=1.
    assert SegmentationEvaluation(true_positives=10, false_positives=0, false_negatives=0).r_val == 1.0


def test_r_value_matches_closed_form_for_a_known_point() -> None:
    # recall = precision = 1/2 so os = 0; then r1 = sqrt((1-1/2)^2 + 0) = 1/2 and r2 = |1/2 - 1| / sqrt(2),
    # giving R = 1 - (1/2 + (1/2)/sqrt(2)) / 2.
    seg = SegmentationEvaluation(true_positives=2, false_positives=2, false_negatives=2)
    expected = 1 - (0.5 + 0.5 / math.sqrt(2)) / 2
    assert seg.os == 0.0
    assert seg.r_val == pytest.approx(expected)


@given(tp=st.integers(1, 1000), fp=counts, fn=counts)
def test_r_value_is_one_iff_perfect(tp: int, fp: int, fn: int) -> None:
    # Sec. 4: R reaches its maximum of 1 only at the target point (no insertions, no deletions).
    seg = SegmentationEvaluation(tp, fp, fn)
    assert (seg.r_val == 1.0) == (fp == 0 and fn == 0)
