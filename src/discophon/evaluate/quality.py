"""Units quality."""

import numpy as np
from xarray import DataArray


def probability_phone_given_unit(coocurrence: DataArray) -> DataArray:
    """P(phone|unit) from a DataArray."""
    coocurrence = coocurrence[:, coocurrence.any(dim="phone")]
    proba = coocurrence / coocurrence.sum(dim="phone")
    most_probable_phones = proba.idxmax(dim="phone")
    units_order = []
    for phone in proba["phone"]:
        indices = np.where(most_probable_phones == phone)[0]
        units_order.extend(indices[np.argsort(proba.sel(phone=phone).values[indices])[::-1]].tolist())
    return proba[:, units_order].rename("P(phone|unit)")


def pnmi(coocurrence: DataArray) -> float:
    """Compute PNMI.

    Args:
        coocurrence: Coocurrence matrix between `units` and the underlying phones, computed with
            [`coocurrence_matrix`][]

    Returns:
        Phone-normalized mutual information (between 0 and 1)
    """
    count = coocurrence.values
    eps = 1e-10
    proba = count / count.sum()
    px, py = proba.sum(axis=1, keepdims=True), proba.sum(axis=0, keepdims=True)
    mutual_info = (proba * np.log(proba / (px @ py + eps) + eps)).sum()
    entropy_x = (-px * np.log(px + eps)).sum()
    return (mutual_info / entropy_x).item()
