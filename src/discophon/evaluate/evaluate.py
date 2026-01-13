"""Phoneme discovery evaluation."""

from typing import TypedDict

from discophon.core import Phones, Units, validate_first_two_arguments_same_keys

from .boundaries import boundary_evaluation
from .per import phoneme_error_rate
from .pnmi import compute_pnmi_and_predict


class PhonemeDiscoveryEvaluation(TypedDict):
    """Output of phoneme discovery evaluation."""

    pnmi: float
    per: float
    f1: float
    r_val: float


@validate_first_two_arguments_same_keys
def phoneme_discovery(
    units: Units,
    phones: Phones,
    *,
    n_units: int,
    n_phonemes: int,
    step_units: int,
    step_phones: int = 10,
) -> PhonemeDiscoveryEvaluation:
    """Full evaluation of phoneme discovery: PNMI, PER, F1 and R-value boundary detection."""
    pnmi, predictions = compute_pnmi_and_predict(
        units,
        phones,
        n_units=n_units,
        n_phonemes=n_phonemes,
        step_units=step_units,
        step_phones=step_phones,
    )
    per = phoneme_error_rate(predictions, phones)
    detection = boundary_evaluation(predictions, phones, step_units=step_units, step_phones=step_phones)
    return {"pnmi": pnmi, "per": per, "f1": detection.f1, "r_val": detection.r_val}
