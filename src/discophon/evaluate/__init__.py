"""Evaluation package for the Phoneme Discovery benchmark."""

from discophon.evaluate.assignment import coocurrence_matrix, get_assignment, relabel_assignment
from discophon.evaluate.discovery import phoneme_discovery
from discophon.evaluate.quality import pnmi, probability_phone_given_unit
from discophon.evaluate.recognition import phone_error_rate
from discophon.evaluate.segmentation import boundary_detection

__all__ = [
    "boundary_detection",
    "coocurrence_matrix",
    "get_assignment",
    "phone_error_rate",
    "phoneme_discovery",
    "pnmi",
    "probability_phone_given_unit",
    "relabel_assignment",
]
