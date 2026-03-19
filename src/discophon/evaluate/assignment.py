import itertools
from collections.abc import Iterable
from typing import Literal, TypedDict

import numpy as np
import polars as pl
from scipy.optimize import linear_sum_assignment
from xarray import DataArray

from discophon.data import Phones, Units
from discophon.validate import validate_first_two_arguments_same_keys

type AssignmentKind = Literal["many-to-one", "one-to-one"]


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
        # if (len(this_phones) - min_len > repeat) or (len(this_units) - min_len > repeat):
        # raise ValueError(f"More than {repeat} tokens of differences between phones and units.")
        data[fileid] = {"phones": this_phones[:min_len], "units": this_units[:min_len]}
    return data


@validate_first_two_arguments_same_keys
def coocurrence_matrix(
    units: Units,
    phones: Phones,
    *,
    n_units: int,
    n_phonemes: int,
    step_units: int,
    step_phones: int,
) -> DataArray:
    """Return a 2D coocurrence matrix of shape (n_phones, n_units).

    Element (i, j) is the number of times the unit j has appeared where the underlying phoneme is i.
    The phonemes are ordered according to the returned dictionary (sorted by frequency).
    """
    n_phonemes_with_sil = n_phonemes + 1
    index, phone_to_index = 0, {}
    phone_indices, unit_indices = [], []
    data = align_units_and_phones(units, phones, step_units=step_units, step_phones=step_phones)
    for phones_and_units in data.values():
        for phone, unit in zip(phones_and_units["phones"], phones_and_units["units"], strict=True):
            if phone not in phone_to_index:
                phone_to_index[phone] = index
                index += 1
            if phone_to_index[phone] >= n_phonemes_with_sil or unit >= n_units:
                raise IndexError
            phone_indices.append(phone_to_index[phone])
            unit_indices.append(unit)
    for missing in range(len(phone_to_index), n_phonemes_with_sil):
        phone_to_index[f"<missing-{missing}>"] = missing

    flattened_indices = np.array(phone_indices) * n_units + np.array(unit_indices)
    count = DataArray(
        np.bincount(flattened_indices, minlength=n_phonemes_with_sil * n_units).reshape(n_phonemes_with_sil, n_units),
        dims=["phone", "unit"],
        coords=[list(phone_to_index.keys()), list(range(n_units))],
        name="Coocurrence Matrix",
    )
    return count.sortby(count.sum(dim="unit"), ascending=False)


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


def mapping_many_to_one(coocurrence: DataArray) -> dict[int, str]:
    """Map each unit to the phoneme that it was associated with the most.

    Many units can be associated to the same phoneme.
    """
    most_frequent = coocurrence.idxmax(dim="phone")
    return dict(
        zip(
            most_frequent.get_index("unit").values.tolist(),
            most_frequent.values.tolist(),
            strict=True,
        )
    )


def mapping_one_to_one(coocurrence: DataArray) -> dict[int, str]:
    phones_idx, units_idx = linear_sum_assignment(coocurrence.values, maximize=True)
    return dict(
        zip(
            coocurrence.get_index("unit").values[units_idx].tolist(),
            coocurrence.get_index("phone").values[phones_idx].tolist(),
            strict=True,
        )
    )


def get_assignment(units: Units, coocurrence: DataArray, *, kind: AssignmentKind) -> Phones:
    match kind:
        case "many-to-one":
            mapping = mapping_many_to_one(coocurrence)
        case "one-to-one":
            mapping = mapping_one_to_one(coocurrence)
        case _:
            raise ValueError(f"Unknown kind: {kind}")
    return {fileid: [mapping[u] for u in this_units] for fileid, this_units in units.items()}
