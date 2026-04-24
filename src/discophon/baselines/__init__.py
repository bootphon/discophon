"""Baseline finetuning."""

from discophon.baselines.hubert import finetune_hubert
from discophon.baselines.spidr import finetune_spidr

__all__ = ["finetune_hubert", "finetune_spidr"]
