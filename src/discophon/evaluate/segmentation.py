"""Phone segmentation."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby

import numpy as np

from discophon.data import STEP_PHONES, STEP_UNITS, Phones
from discophon.validate import validate_first_two_arguments_same_keys


class NoBoundariesError(ZeroDivisionError):
    """Raised when a segmentation metric is undefined because there are no boundaries to score."""

    def __init__(self, which: str) -> None:
        super().__init__(f"Segmentation metric is undefined: no {which} boundaries to score.")


@dataclass(frozen=True)
class SegmentationEvaluation:
    """Container for segmentation results. Target metrics are available as properties."""

    true_positives: int
    false_positives: int
    false_negatives: int

    @cached_property
    def recall(self) -> float:
        """Recall."""
        if self.true_positives + self.false_negatives == 0:
            raise NoBoundariesError("gold")
        return self.true_positives / (self.true_positives + self.false_negatives)

    @cached_property
    def precision(self) -> float:
        """Precision."""
        if self.true_positives + self.false_positives == 0:
            raise NoBoundariesError("predicted")
        return self.true_positives / (self.true_positives + self.false_positives)

    @cached_property
    def f1(self) -> float:
        """F1 score."""
        if 2 * self.true_positives + self.false_positives + self.false_negatives == 0:
            raise NoBoundariesError("gold or predicted")
        return 2 * self.true_positives / (2 * self.true_positives + self.false_positives + self.false_negatives)

    @cached_property
    def os(self) -> float:
        """Over segmentation. Equivalent to recall / precision - 1."""
        if self.true_positives + self.false_negatives == 0:
            raise NoBoundariesError("gold")
        return (self.true_positives + self.false_positives) / (self.true_positives + self.false_negatives) - 1

    @cached_property
    def r_val(self) -> float:
        """R-value from (Rasanen et al., 2009)."""
        r1 = math.sqrt((1 - self.recall) ** 2 + self.os**2)
        r2 = abs(-self.os + self.recall - 1) / math.sqrt(2)
        return 1 - (r1 + r2) / 2

    def describe(self) -> str:
        """All metrics."""
        return "\n".join(
            [
                f"True positives: {self.true_positives}",
                f"False positives: {self.false_positives}",
                f"False negatives: {self.false_negatives}",
                f"Precision: {self.precision:.2%}",
                f"Recall: {self.recall:.2%}",
                f"F1: {self.f1:.2%}",
                f"OS: {self.os:.2%}",
                f"R-val: {self.r_val:.2%}",
            ]
        )

    def __add__(self, other: object) -> "SegmentationEvaluation":
        if not isinstance(other, SegmentationEvaluation):
            raise NotImplementedError
        return SegmentationEvaluation(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )


class Boundaries:
    """Segmentation boundaries."""

    def __init__(self, times_in_ms: Iterable[int]) -> None:
        self._times = np.unique(np.fromiter(times_in_ms, dtype=np.int64))
        self._times.setflags(write=False)

    def __len__(self) -> int:
        return len(self._times)

    def __str__(self) -> str:
        return "[ " + "  ".join([f"{t}s" for t in (self.times / 1000)]) + " ]"

    def tolerance(self, margin_in_ms: int) -> np.ndarray[tuple[int, int], np.dtype[np.int64]]:
        """Tolerance windows for detection for each boundary.

        The window is +/- margin_in_ms around each boundary. Overlapping (or touching) windows are split at the
        midpoint between the two boundaries, which goes to the earlier window while the later one starts just
        after. The resulting windows are disjoint, so a single predicted boundary cannot be counted as a hit for
        two gold boundaries. Follows the procedure from (Rasanen et al., 2009, sec. 2.3).
        """
        lower = (self.times - margin_in_ms).clip(0)
        upper = self.times + margin_in_ms
        midpoints = (self.times[:-1] + self.times[1:]) // 2
        upper[:-1] = np.minimum(upper[:-1], midpoints)
        lower[1:] = np.maximum(lower[1:], midpoints + 1)
        return np.stack([lower, upper], axis=1)

    @property
    def times(self) -> np.ndarray[tuple[int], np.dtype[np.int64]]:
        """Times (in ms) associated to the boundaries."""
        return self._times

    @classmethod
    def from_tokens(cls, tokens: Iterable, step_in_ms: int) -> "Boundaries":
        """Build boundaries from the sequence of non-deduplicated tokens.

        Each boundary corresponds to the transition between two groups of different tokens.
        """
        count = [len(list(group)) for _, group in groupby(tokens)]
        times = (np.array(count, dtype=np.int64).cumsum() * step_in_ms)[:-1]
        return Boundaries(times)


def compare_boundaries(gold: Boundaries, prediction: Boundaries, *, margin_in_ms: int) -> SegmentationEvaluation:
    """Evaluate the detection of the gold boundaries by the prediction (search region method).

    A tolerance window is placed around each *gold* boundary (see [`Boundaries.tolerance`][]). A gold window
    holding at least one predicted boundary is a hit; any further predictions in it are insertions, and gold
    windows left empty are deletions. The argument order therefore matters: `compare_boundaries(gold, prediction)`
    is not the swap of `compare_boundaries(prediction, gold)` in general (Rasanen et al., 2009, sec. 2.3).
    """
    windows = gold.tolerance(margin_in_ms)
    starts = windows[:, 0][:, np.newaxis]
    ends = windows[:, 1][:, np.newaxis]
    detected = ((prediction.times >= starts) & (prediction.times <= ends)).any(axis=1)  # Broadcast and then reduce
    true_positives = detected.sum().item()
    return SegmentationEvaluation(
        true_positives=true_positives,
        false_positives=len(prediction.times) - true_positives,
        false_negatives=len(gold.times) - true_positives,
    )


@validate_first_two_arguments_same_keys
def phone_segmentation(
    predicted_phones_from_units: Phones,
    gold_phones: Phones,
    *,
    margin_in_ms: int = 20,
    step_units: int = STEP_UNITS,
    step_phones: int = STEP_PHONES,
) -> SegmentationEvaluation:
    """Phone segmentation evaluation.

    Arguments:
        predicted_phones_from_units: Predicted phones obtained with
            [`phone_assignments`][discophon.evaluate.phone_assignments]
        gold_phones: Gold phone annotations
        margin_in_ms: Left and right margin around each gold boundaries (in ms).
            Predicted boundaries that fall in the resulting windows are considered correct.
            If two windows overlap, they are cut to the midpoint.
        step_units: Step between consecutive units (in ms)
        step_phones: Step between consecutive phones (in ms)

    Returns:
        Instance of a dataclass containing the segmentation results in attributes `recall`, `precision`, `f1`,
            `os`, and `r_val`. Use its `describe` method to get a summary of the segmentation evaluation.
    """
    return sum(
        (
            compare_boundaries(
                Boundaries.from_tokens(gold_phones[fileid], step_phones),
                Boundaries.from_tokens(predicted_phones_from_units[fileid], step_units),
                margin_in_ms=margin_in_ms,
            )
            for fileid in gold_phones
        ),
        SegmentationEvaluation(0, 0, 0),
    )
