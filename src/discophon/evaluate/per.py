"""Phoneme recognition."""

from collections.abc import Iterable, Sequence
from itertools import groupby

import numba
import numpy as np
from joblib import Parallel, delayed

from discophon.core import Phones, validate_first_two_arguments_same_keys


def deduplicate[T](seq: Iterable[T]) -> list[T]:
    """Deduplicate consecutive values."""
    return [key for key, _ in groupby(seq)]


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


@validate_first_two_arguments_same_keys
def phoneme_error_rate(hypothesis: Phones, target: Phones, *, n_jobs: int = -1) -> float:
    """Phoneme error rate: total edit distances divided by the length of the target corpus."""
    results = Parallel(n_jobs=n_jobs)(
        delayed(lambda x, y: (edit_distance(x, y), len(y)))(
            deduplicate(hypothesis[fileid]), deduplicate(target[fileid])
        )
        for fileid in hypothesis
    )
    edit_distances, lengths = zip(*results, strict=True)
    return sum(edit_distances) / sum(lengths)
