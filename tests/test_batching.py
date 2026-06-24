"""Check that batched feature extraction matches individual forward passes."""

import datetime

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("torch")
pytest.importorskip("minimal_hubert")
pytest.importorskip("spidr")

import torch
from minimal_hubert import HuBERT
from spidr.models import SpidR, spidr_base
from torch import nn
from torch.nn import functional as F
from torch.testing import assert_close, make_tensor

from discophon.baselines.utils import collate_fn

DEVICES = ["cpu", *(["cuda"] if torch.cuda.is_available() else [])]


@pytest.fixture(scope="module")
def spidr_model() -> SpidR:
    return spidr_base(pretrained=False).eval()


@pytest.fixture(scope="module")
def hubert_model() -> HuBERT:
    return HuBERT().eval()


def assert_batched_matches_individual(model: nn.Module, lengths: list[int], device: str) -> None:
    model = model.to(device)
    waveforms = [F.layer_norm(make_tensor(n, dtype=torch.float32, device=device), (n,)) for n in lengths]
    with torch.inference_mode():
        individual = [model.get_intermediate_outputs(w.unsqueeze(0)) for w in waveforms]
        wavs, attn_mask, feat_lengths = collate_fn(waveforms)
        batched = model.get_intermediate_outputs(wavs.to(device), attention_mask=attn_mask.to(device))
    for i, single in enumerate(individual):
        valid = int(feat_lengths[i])
        assert single[0].size(1) == valid
        for layer in range(len(batched)):
            assert_close(batched[layer][i, :valid], single[layer][0])


@pytest.mark.slow
@pytest.mark.parametrize("device", DEVICES)
@settings(deadline=datetime.timedelta(seconds=30))
@given(lengths=st.lists(st.integers(min_value=8_000, max_value=32_000), min_size=2, max_size=4))
def test_spidr_batched_matches_individual(spidr_model: SpidR, device: str, lengths: list[int]) -> None:
    assert_batched_matches_individual(spidr_model, lengths, device)


@pytest.mark.slow
@pytest.mark.xfail(
    reason="HuBERT base uses extractor_mode='group_norm', whose GroupNorm normalizes over the time "
    "axis; zero padding changes the statistics so a padded batch does not match individual passes.",
)
@pytest.mark.parametrize("device", DEVICES)
@settings(deadline=datetime.timedelta(seconds=30))
@given(lengths=st.lists(st.integers(min_value=8_000, max_value=32_000), min_size=2, max_size=4))
def test_hubert_batched_matches_individual(hubert_model: HuBERT, device: str, lengths: list[int]) -> None:
    assert_batched_matches_individual(hubert_model, lengths, device)
