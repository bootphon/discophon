import math
from collections.abc import Iterable
from dataclasses import dataclass
from functools import cached_property
from itertools import groupby

import numpy as np

from .utils import Phones, validate_same_keys


@dataclass(frozen=True)
class DetectionResult:
    true_positives: int
    false_positives: int
    false_negatives: int

    @cached_property
    def recall(self) -> float:
        return self.true_positives / (self.true_positives + self.false_negatives)

    @cached_property
    def precision(self) -> float:
        return self.true_positives / (self.true_positives + self.false_positives)

    @cached_property
    def f1(self) -> float:
        return 2 * self.true_positives / (2 * self.true_positives + self.false_positives + self.false_negatives)

    @cached_property
    def os(self) -> float:
        return self.recall / self.precision - 1

    @cached_property
    def r_val(self) -> float:
        r1 = math.sqrt((1 - self.recall) ** 2 + self.os**2)
        r2 = abs(-self.os + self.recall - 1) / math.sqrt(2)
        return 1 - (r1 + r2) / 2

    def describe(self) -> str:
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

    def __add__(self, other: object) -> "DetectionResult":
        if not isinstance(other, DetectionResult):
            raise NotImplementedError
        return DetectionResult(
            true_positives=self.true_positives + other.true_positives,
            false_positives=self.false_positives + other.false_positives,
            false_negatives=self.false_negatives + other.false_negatives,
        )


def times_from_tokens(tokens: Iterable, step: int) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
    count = [len(list(group)) for _, group in groupby(tokens)]
    return (np.array(count, dtype=np.int64).cumsum() * step)[:-1]


class Boundaries:
    def __init__(self, times_in_ms: Iterable[int]) -> None:
        self._times = np.array(times_in_ms, dtype=np.int64)
        self._times.sort()
        self._times.setflags(write=0)

    def __len__(self) -> int:
        return len(self._times)

    def __str__(self) -> str:
        return "[ " + "  ".join([f"{t}s" for t in (self.times / 1000)]) + " ]"

    def tolerance(self, margin_in_ms: int) -> np.ndarray[tuple[int, int], np.dtype[np.float64]]:
        windows = np.vstack([self.times - margin_in_ms, self.times + margin_in_ms]).clip(0).T
        overlaps = windows[:-1, 1] > windows[1:, 0]
        midpoints = (windows[:-1, 1] + windows[1:, 0]) // 2
        windows[:-1, 1] = np.where(overlaps, midpoints, windows[:-1, 1])
        windows[1:, 0] = np.where(overlaps, midpoints, windows[1:, 0])
        return windows

    @property
    def times(self) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        return self._times

    @classmethod
    def from_tokens(cls, tokens: Iterable, step: int) -> "Boundaries":
        return Boundaries(times_from_tokens(tokens, step))


def boundary_detection(gold: Boundaries, prediction: Boundaries, *, margin_in_ms: int = 20) -> DetectionResult:
    windows = gold.tolerance(margin_in_ms)
    starts = windows[:, 0][:, np.newaxis]
    ends = windows[:, 1][:, np.newaxis]
    detected = ((prediction.times >= starts) & (prediction.times <= ends)).any(axis=1)  # Broadcast and then reduce
    true_positives = detected.sum().item()
    return DetectionResult(
        true_positives=true_positives,
        false_positives=len(prediction.times) - true_positives,
        false_negatives=len(gold.times) - true_positives,
    )


@validate_same_keys
def evaluate_boundaries(
    predicted_phones_from_units: Phones,
    gold_phones: Phones,
    *,
    step_units: int,
    step_phones: int,
    margin_in_ms: int = 20,
) -> DetectionResult:
    if set(predicted_phones_from_units) != set(gold_phones):
        raise ValueError
    return sum(
        (
            boundary_detection(
                Boundaries.from_tokens(gold_phones[fileid], step=step_phones),
                Boundaries.from_tokens(predicted_phones_from_units[fileid], step=step_units),
                margin_in_ms=margin_in_ms,
            )
            for fileid in gold_phones
        ),
        DetectionResult(0, 0, 0),
    )
