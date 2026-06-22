"""Phone recognition."""

from collections.abc import Iterable, Sequence
from itertools import groupby

import numba
import numpy as np
from joblib import Parallel, delayed

from discophon.data import Phones
from discophon.validate import validate_first_two_arguments_same_keys


def deduplicate[T](seq: Iterable[T]) -> list[T]:
    """Deduplicate consecutive values into a numba typed list (so it can be passed to `edit_distance`)."""
    deduplicated = [key for key, _ in groupby(seq)]
    if len(deduplicated) == 0:
        raise ValueError("Empty sequence found while deduplicating")
    return deduplicated


@numba.jit(nopython=True, nogil=True)
def edit_distance[T](hypothesis: Sequence[T], target: Sequence[T]) -> int:
    """Edit distance.

    Based on the torchaudio implementation:
    https://github.com/pytorch/audio/blob/ad5816f0eee1c873df1b7d371c69f1f811a89387/src/torchaudio/functional/functional.py#L1493
    """
    dold = np.arange(len(target) + 1)
    dnew = np.zeros_like(dold)
    for i in range(1, len(hypothesis) + 1):
        dnew[0] = i
        for j in range(1, len(target) + 1):
            if hypothesis[i - 1] == target[j - 1]:
                dnew[j] = dold[j - 1]
            else:
                substitution = dold[j - 1] + 1
                insertion = dnew[j - 1] + 1
                deletion = dold[j] + 1
                dnew[j] = min(substitution, insertion, deletion)
        dnew, dold = dold, dnew
    return dold[-1].item()


def _edit_distance_and_length(predicted: Sequence[str], gold: Sequence[str]) -> tuple[int, int]:
    hypothesis, target = deduplicate(predicted), deduplicate(gold)
    return edit_distance(hypothesis, target), len(target)


@validate_first_two_arguments_same_keys
def phone_error_rate(predicted_phones_from_units: Phones, gold_phones: Phones, *, n_jobs: int = -1) -> float:
    """Phone error rate.

    Total edit distances divided by the total length of the target annotations.

    Arguments:
        predicted_phones_from_units: Predicted phones obtained with
            [`phone_assignments`][discophon.evaluate.phone_assignments]
        gold_phones: Gold phone annotations
        n_jobs: The maximum number of concurrently runnings jobs to be passed to [`joblib.Parallel`][]

    Returns:
        Phone error rate. Multiply it by 100 to get a percentage.
    """
    results = Parallel(n_jobs=n_jobs)(
        delayed(_edit_distance_and_length)(predicted_phones_from_units[fileid], gold_phones[fileid])
        for fileid in predicted_phones_from_units
    )
    edit_distances, lengths = zip(*results, strict=True)
    return sum(edit_distances) / sum(lengths)
