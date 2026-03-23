"""Phoneme discovery evaluation."""

from typing import TypedDict

from discophon.data import STEP_PHONES, Phones, Units
from discophon.evaluate.assignment import AssignmentKind, coocurrence_matrix, get_assignment
from discophon.evaluate.quality import pnmi
from discophon.evaluate.recognition import phone_error_rate
from discophon.evaluate.segmentation import boundary_detection
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
    n_units: int,
    n_phonemes: int,
    step_units: int,
    kind: AssignmentKind,
) -> PhonemeDiscoveryEvaluation:
    """Full evaluation of phoneme discovery: PNMI, PER, F1 and R-value boundary detection."""
    coocurrence = coocurrence_matrix(
        units,
        phones,
        n_units=n_units,
        n_phonemes=n_phonemes,
        step_units=step_units,
        step_phones=STEP_PHONES,
    )
    assignment = get_assignment(units, coocurrence, kind=kind)
    per = phone_error_rate(assignment, phones)
    detection = boundary_detection(assignment, phones, step_units=step_units, step_phones=STEP_PHONES)
    return {"pnmi": pnmi(coocurrence), "per": per, "f1": detection.f1, "r_val": detection.r_val}
