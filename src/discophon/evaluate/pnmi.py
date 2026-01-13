"""Assignment and mutual information."""

import itertools
from collections.abc import Iterable
from typing import TypedDict

import numpy as np
import polars as pl
from xarray import DataArray

from discophon.core import Phones, Units, validate_first_two_arguments_same_keys


class UnitsAndPhones(TypedDict):
    """Dictionary mapping filenames to the corresponding predicted units and gold phonemes."""

    units: list[int]
    phones: list[str]


@validate_first_two_arguments_same_keys
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
            raise ValueError(f"More than {repeat} tokens of differences between phones and units.")
        data[fileid] = {"phones": this_phones[:min_len], "units": this_units[:min_len]}
    return data


def contingency_table(
    units: Units,
    phones: Phones,
    *,
    n_units: int,
    n_phonemes: int,
    step_units: int,
    step_phones: int,
) -> DataArray:
    """Return a 2D contingency table of shape (n_phones, n_units).

    Element (i, j) is the number of times the unit j has appeared where the underlying phoneme is i.
    The phonemes are ordered according to the returned dictionary (sorted by frequency).
    """
    index, phone_to_index = 0, {}
    phone_indices, unit_indices = [], []
    data = align_units_and_phones(units, phones, step_units=step_units, step_phones=step_phones)
    for phones_and_units in data.values():
        for phone, unit in zip(phones_and_units["phones"], phones_and_units["units"], strict=True):
            if phone not in phone_to_index:
                phone_to_index[phone] = index
                index += 1
            if phone_to_index[phone] >= n_phonemes or unit >= n_units:
                raise IndexError
            phone_indices.append(phone_to_index[phone])
            unit_indices.append(unit)
    for missing in range(len(phone_to_index), n_phonemes):
        phone_to_index[f"<missing-{missing}>"] = missing

    flattened_indices = np.array(phone_indices) * n_units + np.array(unit_indices)
    count = DataArray(
        np.bincount(flattened_indices, minlength=n_phonemes * n_units).reshape(n_phonemes, n_units),
        dims=["phone", "unit"],
        coords=[list(phone_to_index.keys()), list(range(n_units))],
        name="Contingency Table",
    )
    return count.sortby(count.sum(axis=1), ascending=False)


def probability_phone_given_unit(count: DataArray) -> DataArray:
    """Return P(phone|unit) as a xarray DataArray."""
    count = count[:, count.any(dim="phone")]
    proba = count / count.sum(dim="phone")
    most_probable_phones = proba.idxmax(dim="phone")
    units_order = []
    for phone in proba["phone"]:
        indices = np.where(most_probable_phones == phone)[0]
        units_order.extend(indices[np.argsort(proba.sel(phone=phone).values[indices])[::-1]].tolist())
    return proba[:, units_order].rename("P(phone|unit)")


def relabel_assignment(assignment: Iterable[int], proba: DataArray) -> DataArray:
    """Relabel the assignment of units to phones according to the most probable phones."""
    c_proba, c_phone, c_unit = str(proba.name), "phone", "unit"
    df_assignment = pl.DataFrame({c_unit: proba[c_unit].to_numpy(), "assignment": np.array(assignment)})
    df_proba = pl.DataFrame(proba.to_dataframe().reset_index()).join(df_assignment, on=c_unit, how="left")
    most_probable = (
        df_proba.group_by("assignment", c_phone, maintain_order=True)
        .agg(pl.col(c_proba).mean())
        .group_by("assignment", maintain_order=True)
        .agg(pl.all().sort_by(c_proba).last())
        .join(
            pl.DataFrame(proba[c_phone].to_numpy(), schema={c_phone: pl.String}).with_row_index(),
            on=c_phone,
        )
        .sort(pl.col("index"), -pl.col(c_proba))
    )
    order = {v: k for k, v in enumerate(most_probable["assignment"].to_list())}
    new_assignment = df_assignment.with_columns(pl.col("assignment").replace_strict(order))
    return DataArray(
        new_assignment["assignment"],
        dims=[c_unit],
        coords=[proba[c_unit]],
        name="assignment",
    )


def pnmi(contingency: np.ndarray[tuple[int, int], np.dtype[np.int64]], *, eps: float = 1e-10) -> float:
    """Phone normalized mutual information, as in (Hsu et al., 2021)."""
    proba = contingency / contingency.sum()
    px, py = proba.sum(axis=1, keepdims=True), proba.sum(axis=0, keepdims=True)
    mutual_info = (proba * np.log(proba / (px @ py + eps) + eps)).sum()
    entropy_x = (-px * np.log(px + eps)).sum()
    return (mutual_info / entropy_x).item()


def mapping_many_to_one(contingency: DataArray) -> dict[int, str]:
    """Map each unit to the phoneme that it was associated with the most.

    Many units can be associated to the same phoneme.
    """
    most_frequent = contingency.idxmax(dim="phone")
    return dict(
        zip(
            most_frequent.get_index("unit").values.tolist(),
            most_frequent.values.tolist(),
            strict=True,
        )
    )


def mapping_one_to_one(
    count: np.ndarray[tuple[int, int], np.dtype[np.int64]],
    phone_order: dict[int, str],
    *,
    unk_template: str = "<unknown-{index}>",
) -> dict[int, str]:
    """Map the most frequent unit to the corresponding phoneme.

    The mapping is one-to-one: each phoneme is assigned to exactly one unit.
    Units that are not assigned to any phoneme are set to unknown.
    """
    most_frequent = count.argmax(axis=0)
    highest_counts = count[most_frequent, np.arange(count.shape[1])]
    assignments = {}
    for phone in np.unique(most_frequent):
        (assigned_units,) = np.where(most_frequent == phone)
        best_unit = assigned_units[np.argmax(highest_counts[assigned_units])]
        assignments[best_unit.item()] = phone_order[phone]
    for unit in range(count.shape[1]):
        if unit not in assignments:
            assignments[unit] = unk_template.format(index=unit)
    return dict(sorted(assignments.items()))


@validate_first_two_arguments_same_keys
def compute_pnmi_and_predict(
    units: Units,
    phones: Phones,
    *,
    n_units: int,
    n_phonemes: int,
    step_units: int,
    step_phones: int,
) -> tuple[float, Phones]:
    """Compute the PNMI and the predicted phoneme transcription using the many-to-one scheme."""
    contingency = contingency_table(
        units,
        phones,
        n_units=n_units,
        n_phonemes=n_phonemes,
        step_units=step_units,
        step_phones=step_phones,
    )
    mapping = mapping_many_to_one(contingency)
    predictions = {fileid: [mapping[u] for u in this_units] for fileid, this_units in units.items()}
    return pnmi(contingency.values), predictions
