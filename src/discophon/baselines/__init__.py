"""Baseline finetuning."""

from discophon.baselines.hubert import (
    extract_hubert_continuous_features,
    extract_hubert_discrete_units,
    finetune_hubert,
)
from discophon.baselines.spidr import extract_spidr_continuous_features, extract_spidr_discrete_units, finetune_spidr

__all__ = [
    "extract_hubert_continuous_features",
    "extract_hubert_discrete_units",
    "extract_spidr_continuous_features",
    "extract_spidr_discrete_units",
    "finetune_hubert",
    "finetune_spidr",
]
