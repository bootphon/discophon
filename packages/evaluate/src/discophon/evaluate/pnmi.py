"""Assignment and mutual information."""

import itertools

import numpy as np

from .utils import Phones, Units, UnitsAndPhones, validate_same_keys


@validate_same_keys
def align_units_and_phones(
    units: Units,
    phones: Phones,
    *,
    step_units: int,
    step_phones: int,
) -> dict[str, UnitsAndPhones]:
    """Align units and phones by repeating each unit step_units // step_phones times.

    This assumes that step_units <= step_phones and they step_phones is a multiple of step_units.
    Allows for a small margin in the end, in case where the final unit is missing.
    """
    repeat = step_units // step_phones
    data = {}
    for fileid, this_phones in phones.items():
        this_units = list(itertools.chain.from_iterable(itertools.repeat(unit, repeat) for unit in units[fileid]))
        min_len = min(len(this_phones), len(this_units))
        if (len(this_phones) - min_len > repeat) or (len(this_units) - min_len > repeat):
            raise ValueError
        data[fileid] = {"phones": this_phones[:min_len], "units": this_units[:min_len]}
    return data


def contingency_table(
    units: Units,
    phones: Phones,
    *,
    n_units: int,
    n_phones: int,
    step_units: int,
    step_phones: int,
) -> tuple[np.ndarray[tuple[int, int], np.dtype[np.int64]], dict[int, str]]:
    """Return a 2D contingency table of shape (n_phones, n_units).

    Element (i, j) is the number of times the unit j has appeared where the underlying phoneme is i.
    The phonemes are ordered according to the returned dictionary (sorted by frequency).
    """
    index, phone_to_index, index_to_phone = 0, {}, {}
    phone_indices, unit_indices = [], []
    data = align_units_and_phones(units, phones, step_units=step_units, step_phones=step_phones)
    for phones_and_units in data.values():
        for phone, unit in zip(phones_and_units["phones"], phones_and_units["units"], strict=True):
            if phone not in phone_to_index:
                phone_to_index[phone] = index
                index_to_phone[index] = phone
                index += 1
            if phone_to_index[phone] >= n_phones or unit >= n_units:
                raise IndexError
            phone_indices.append(phone_to_index[phone])
            unit_indices.append(unit)

    flattened_indices = np.array(phone_indices) * n_units + np.array(unit_indices)
    count = np.bincount(flattened_indices, minlength=n_phones * n_units).reshape(n_phones, n_units)
    most_frequent_phones = np.argsort(count.sum(axis=1))[::-1]
    index_to_phone = {v: k for k, v in phone_to_index.items()}
    phone_order = {k: index_to_phone[i.item()] for k, i in enumerate(most_frequent_phones)}
    return count[most_frequent_phones], phone_order


def pnmi(count: np.ndarray[tuple[int, int], np.dtype[np.int64]], *, eps: float = 1e-10) -> float:
    """Phone normalized mutual information, as in (Hsu et al., 2021)."""
    proba = count / count.sum()
    px, py = proba.sum(axis=1, keepdims=True), proba.sum(axis=0, keepdims=True)
    mutual_info = (proba * np.log(proba / (px @ py + eps) + eps)).sum()
    entropy_x = (-px * np.log(px + eps)).sum()
    return (mutual_info / entropy_x).item()


def mapping_many_to_one(
    count: np.ndarray[tuple[int, int], np.dtype[np.int64]],
    phone_order: dict[int, str],
) -> dict[int, str]:
    """Map each unit to the phoneme that it was associated with the most.

    Many units can be associated to the same phoneme.
    """
    most_frequent = count.argmax(axis=0).tolist()
    return {k: phone_order[p] for k, p in enumerate(most_frequent)}


@validate_same_keys
def evaluate_pnmi_and_predict(
    units: Units,
    phones: Phones,
    *,
    n_units: int,
    n_phones: int,
    step_units: int,
    step_phones: int,
) -> tuple[float, Phones]:
    """Compute the PNMI and the predicted phoneme transcription using the many-to-one scheme."""
    count, phone_order = contingency_table(
        units,
        phones,
        n_units=n_units,
        n_phones=n_phones,
        step_units=step_units,
        step_phones=step_phones,
    )
    mapping = mapping_many_to_one(count, phone_order)
    predictions = {fileid: [mapping[u] for u in this_units] for fileid, this_units in units.items()}
    return pnmi(count), predictions
