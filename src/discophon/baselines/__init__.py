"""Baseline finetuning."""

from discophon.baselines.hubert import (
    extract_hubert_continuous_features,
    extract_hubert_discrete_units,
    finetune_hubert,
    validate_all_hubert_checkpoints,
)
from discophon.baselines.spidr import (
    extract_spidr_continuous_features,
    extract_spidr_discrete_units,
    finetune_spidr,
    validate_all_spidr_checkpoints,
)

__all__ = [
    "extract_hubert_continuous_features",
    "extract_hubert_discrete_units",
    "extract_spidr_continuous_features",
    "extract_spidr_discrete_units",
    "finetune_hubert",
    "finetune_spidr",
    "validate_all_hubert_checkpoints",
    "validate_all_spidr_checkpoints",
]
