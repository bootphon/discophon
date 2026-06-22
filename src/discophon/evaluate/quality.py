"""Units quality."""

import numpy as np
from xarray import DataArray


def probability_phone_given_unit(cooccurrence: DataArray) -> DataArray:
    """P(phone|unit) from a DataArray."""
    cooccurrence = cooccurrence[:, cooccurrence.any(dim="phone")]
    proba = cooccurrence / cooccurrence.sum(dim="phone")
    most_probable_phones = proba.idxmax(dim="phone")
    units_order = []
    for phone in proba["phone"]:
        indices = np.where(most_probable_phones == phone)[0]
        units_order.extend(indices[np.argsort(proba.sel(phone=phone).values[indices])[::-1]].tolist())
    return proba[:, units_order].rename("P(phone|unit)")


def pnmi(cooccurrence: DataArray) -> float:
    """Compute PNMI.

    The phone-normalized mutual information is the mutual information between phones and units divided
    by the phone entropy.

    Both quantities are summed over their non-zero entries only, so the identity
    ``0 * log(0) = 0`` holds exactly and the result is guaranteed to lie in `[0, 1]`.
    Degenerate inputs have no information to normalize: if the matrix is empty or a single phone carries
    all the mass (the phone entropy is zero), PNMI is defined to be `0.0`.

    Args:
        cooccurrence: Cooccurrence matrix between `units` and the underlying phones, computed with
            [`cooccurrence_matrix`][discophon.evaluate.cooccurrence_matrix]

    Returns:
        Phone-normalized mutual information (between 0 and 1)
    """
    count = cooccurrence.values
    total = count.sum()
    if total == 0 or (count.sum(axis=1) > 0).sum() <= 1:
        return 0.0
    proba = count / total
    px, py = proba.sum(axis=1, keepdims=True), proba.sum(axis=0, keepdims=True)
    joint = proba[proba > 0]
    independent = (px @ py)[proba > 0]
    mutual_info = (joint * np.log(joint / independent)).sum()
    px_positive = px[px > 0]
    entropy_x = -(px_positive * np.log(px_positive)).sum()
    return (mutual_info / entropy_x).clip(0, 1).item()
