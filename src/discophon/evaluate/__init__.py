"""DiscoPhon evaluation module."""

from discophon.evaluate.assignment import coocurrence_matrix, phone_assignments
from discophon.evaluate.discovery import phoneme_discovery
from discophon.evaluate.quality import pnmi
from discophon.evaluate.recognition import phone_error_rate
from discophon.evaluate.segmentation import phone_segmentation

__all__ = [
    "coocurrence_matrix",
    "phone_assignments",
    "phone_error_rate",
    "phone_segmentation",
    "phoneme_discovery",
    "pnmi",
]
