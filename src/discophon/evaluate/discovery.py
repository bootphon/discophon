"""Phoneme discovery evaluation."""

from typing import Literal, TypedDict

from discophon.data import STEP_PHONES, STEP_UNITS, Phones, Units
from discophon.evaluate.assignment import coocurrence_matrix, phone_assignments
from discophon.evaluate.quality import pnmi
from discophon.evaluate.recognition import phone_error_rate
from discophon.evaluate.segmentation import phone_segmentation
from discophon.validate import validate_first_two_arguments_same_keys


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
    kind: Literal["many-to-one", "one-to-one"],
    n_units: int,
    n_phonemes: int,
    step_units: int = STEP_UNITS,
    step_phones: int = STEP_PHONES,
) -> PhonemeDiscoveryEvaluation:
    """Full evaluation of phoneme discovery: PNMI, PER, F1 and R-value boundary detection.

    Args:
        units: Predicted discrete units
        phones: Gold phone annotations
        kind: Kind of assignment
        n_units: Number of distinct discrete units in the evaluated system
        n_phonemes: Number of phonemes in the language under consideration
        step_units: Step between consecutive units (in ms)
        step_phones: Step between consecutive phones (in ms)
        step_units: Step between consecutive units (in ms)

    Returns:
        Phoneme discovery results in a dictionary with keys `"pnmi"`, `"per"`, `"f1"`, and `"r_val"`.
    """
    coocurrence = coocurrence_matrix(
        units,
        phones,
        n_units=n_units,
        n_phonemes=n_phonemes,
        step_units=step_units,
        step_phones=STEP_PHONES,
    )
    assignment = phone_assignments(units, coocurrence, kind=kind)
    per = phone_error_rate(assignment, phones)
    detection = phone_segmentation(assignment, phones, step_units=step_units, step_phones=step_phones)
    return {"pnmi": pnmi(coocurrence), "per": per, "f1": detection.f1, "r_val": detection.r_val}
