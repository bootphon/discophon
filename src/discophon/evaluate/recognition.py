"""Phone recognition."""

from collections.abc import Iterable, Sequence
from itertools import chain, groupby

import numba
import numpy as np

from discophon.data import Phones
from discophon.validate import validate_first_two_arguments_same_keys


def deduplicate[T](seq: Iterable[T]) -> list[T]:
    """Deduplicate consecutive values."""
    deduplicated = [key for key, _ in groupby(seq)]
    if len(deduplicated) == 0:
        raise ValueError("Empty sequence found while deduplicating")
    return deduplicated


@numba.njit(nogil=True, cache=True)
def edit_distance(
    hypothesis: np.ndarray[tuple[int], np.dtype[np.int64]],
    target: np.ndarray[tuple[int], np.dtype[np.int64]],
) -> int:  # pragma: no cover
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
    index = {phone: i for i, phone in enumerate(chain(hypothesis, target))}
    hypothesis_ids = np.fromiter((index[phone] for phone in hypothesis), dtype=np.int64)
    target_ids = np.fromiter((index[phone] for phone in target), dtype=np.int64)
    return edit_distance(hypothesis_ids, target_ids), len(target)


@validate_first_two_arguments_same_keys
def phone_error_rate(predicted_phones_from_units: Phones, gold_phones: Phones) -> float:
    """Phone error rate.

    Total edit distances divided by the total length of the target annotations.

    Arguments:
        predicted_phones_from_units: Predicted phones obtained with
            [`phone_assignments`][discophon.evaluate.phone_assignments]
        gold_phones: Gold phone annotations

    Returns:
        Phone error rate. Multiply it by 100 to get a percentage.

    Raises:
        ValueError: If there are no files, or if some files have an empty predicted or gold sequence.

    """
    if not gold_phones:
        raise ValueError("No files to evaluate: the predicted and gold phones are empty.")
    if empty := sorted(f for f in gold_phones if not gold_phones[f] or not predicted_phones_from_units[f]):
        raise ValueError(f"Empty predicted or gold sequences for {len(empty)} files, such as {empty[:5]}.")
    total_distance, total_length = 0, 0
    for fileid, gold in gold_phones.items():
        distance, length = _edit_distance_and_length(predicted_phones_from_units[fileid], gold)
        total_distance += distance
        total_length += length
    return total_distance / total_length
