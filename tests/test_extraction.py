"""Tests for the extraction of units and features from the baselines, with a fake model on CPU."""

import functools
import json
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, cast
from unittest.mock import MagicMock

import joblib
import numpy as np
import polars as pl
import pytest
import soundfile as sf

pytest.importorskip("minimal_hubert")
pytest.importorskip("spidr")

import torch
from spidr.config import DEFAULT_CONV_LAYER_CONFIG
from spidr.data.dataset import conv_length

from discophon.baselines import extract, hubert, spidr
from discophon.baselines.extract import cli as extract_cli
from discophon.baselines.utils import build_inference_dataloader
from discophon.data import SAMPLE_RATE, manifest_filename, units_filename
from discophon.languages import all_languages, get_language

if TYPE_CHECKING:
    from sklearn.cluster import MiniBatchKMeans

GERMAN = get_language("deu")
NUM_SAMPLES = {"a": 4_000, "b": 9_000, "c": 5_600, "d": 12_000, "e": 7_000}
N_LAYERS, N_CODEBOOKS, N_CODES = 4, 2, 5


class FakeModel:
    """Stands for HuBERT and SpidR: the features of a frame only depend on its file, its position and its layer.

    The first feature is the first sample of the file, and the second one is the frame index plus the layer (from 1).
    """

    def __init__(self) -> None:
        self.encoder = self.student = SimpleNamespace(layers=[None] * N_LAYERS)
        self.num_codebooks = N_CODEBOOKS
        self.batch_sizes: list[int] = []

    def eval(self) -> "FakeModel":
        """Do nothing."""
        return self

    def cuda(self) -> "FakeModel":
        """Do nothing."""
        return self

    def get_intermediate_outputs(
        self, waveforms: torch.Tensor, *, attention_mask: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """Features of each layer, with shape (batch, frames, 2)."""
        if attention_mask is not None:
            assert attention_mask.size(0) == waveforms.size(0) > 1
        self.batch_sizes.append(waveforms.size(0))
        n_frames = int(conv_length(DEFAULT_CONV_LAYER_CONFIG, torch.tensor([waveforms.size(1)]))[0])
        first, time = waveforms[:, :1].expand(-1, n_frames), torch.arange(n_frames).expand(waveforms.size(0), -1)
        return [torch.stack([first, time + layer + 1], dim=-1) for layer in range(N_LAYERS)]

    def get_codebooks(self, waveforms: torch.Tensor, *, attention_mask: torch.Tensor | None = None) -> list:
        """One-hot logits of the last layers, with the second feature modulo the number of codes as argmax."""
        features = self.get_intermediate_outputs(waveforms, attention_mask=attention_mask)
        return [None] * (N_LAYERS - N_CODEBOOKS) + [
            torch.nn.functional.one_hot(f[..., 1].long() % N_CODES, N_CODES).float() for f in features[-N_CODEBOOKS:]
        ]


class FakeKMeans:
    """K-means whose clusters are the second feature."""

    @staticmethod
    def predict(features: np.ndarray) -> np.ndarray:
        """Cluster of each frame."""
        return features[:, 1].astype(np.int64)


def n_frames(fileid: str) -> int:
    return int(conv_length(DEFAULT_CONV_LAYER_CONFIG, torch.tensor([NUM_SAMPLES[fileid]]))[0])


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    (root / "manifest").mkdir(parents=True)
    (root / "audio" / "deu" / "dev").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for fileid, num_samples in NUM_SAMPLES.items():
        sf.write(root / "audio" / "deu" / "dev" / f"{fileid}.wav", rng.uniform(-0.5, 0.5, num_samples), SAMPLE_RATE)
    manifest = pl.DataFrame({"fileid": list(NUM_SAMPLES), "num_samples": list(NUM_SAMPLES.values())})
    manifest.write_csv(root / "manifest" / manifest_filename(GERMAN, "dev"))
    return root


@pytest.fixture
def model(monkeypatch: pytest.MonkeyPatch) -> FakeModel:
    model = FakeModel()
    monkeypatch.setattr(hubert.HuBERT, "from_pretrained", lambda *_: model)
    monkeypatch.setattr(spidr, "build_model", lambda **_: model)
    monkeypatch.setattr(torch.Tensor, "cuda", lambda self, *_, **__: self)
    for module in (hubert, spidr):
        loader = functools.partial(build_inference_dataloader, num_workers=0)
        monkeypatch.setattr(module, "build_inference_dataloader", loader)
    return model


def read_units(path: Path) -> dict[str, list[int]]:
    entries = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == len({entry["file"] for entry in entries})  # each file only once
    return {entry["file"]: entry["units"] for entry in entries}


def read_features(root: Path) -> dict[str, torch.Tensor]:
    assert not list(root.rglob("*.part"))
    return {str(p.relative_to(root)): torch.load(p) for p in sorted(root.rglob("*.pt"))}


@pytest.mark.parametrize("layers", [[2, 4], None])
def test_hubert_discrete_units(tmp_path: Path, dataset: Path, model: FakeModel, layers: list[int] | None) -> None:
    kmeans = cast("dict[int, MiniBatchKMeans]", {layer: FakeKMeans() for layer in (2, 4)})
    hubert.extract_hubert_discrete_units(dataset, tmp_path / "units", "deu", "dev", "it2.pt", kmeans, layers=layers)
    assert sorted(p.parent.name for p in (tmp_path / "units").rglob("*.jsonl")) == ["2", "4"]  # layers with K-means
    for layer in (2, 4):
        units = read_units(tmp_path / "units" / str(layer) / units_filename(GERMAN, "dev"))
        assert units == {fileid: [t + layer for t in range(n_frames(fileid))] for fileid in NUM_SAMPLES}
    assert model.batch_sizes == [1] * len(NUM_SAMPLES)  # HuBERT is not batch invariant


@pytest.mark.usefixtures("model")
@pytest.mark.parametrize(("layers", "kmeans_layers", "match"), [([1, 2], [2], "No K-means"), (None, [2, 9], "9")])
def test_hubert_discrete_units_rejects_layers_without_kmeans(
    tmp_path: Path, dataset: Path, layers: list[int] | None, kmeans_layers: list[int], match: str
) -> None:
    kmeans = cast("dict[int, MiniBatchKMeans]", dict.fromkeys(kmeans_layers, FakeKMeans()))
    with pytest.raises(ValueError, match=match):
        hubert.extract_hubert_discrete_units(
            dataset, tmp_path / "units", "deu", "dev", "it2.pt", kmeans, layers=layers
        )
    assert not (tmp_path / "units").exists()


def test_hubert_continuous_features(tmp_path: Path, dataset: Path, model: FakeModel) -> None:
    hubert.extract_hubert_continuous_features(dataset, tmp_path / "features", "deu", "dev", "it2.pt", layers=[1, 3])
    features = read_features(tmp_path / "features")
    assert sorted(features) == [f"{layer}/deu/dev/{fileid}.pt" for layer in (1, 3) for fileid in NUM_SAMPLES]
    for layer in (1, 3):
        for fileid in NUM_SAMPLES:
            frames = features[f"{layer}/deu/dev/{fileid}.pt"]
            assert frames.shape == (n_frames(fileid), 2)
            assert frames[:, 1].tolist() == [t + layer for t in range(n_frames(fileid))]
    assert model.batch_sizes == [1] * len(NUM_SAMPLES)


def test_spidr_discrete_units_with_batches(tmp_path: Path, dataset: Path, model: FakeModel) -> None:
    spidr.extract_spidr_discrete_units(dataset, tmp_path / "single", "deu", "dev", "spidr.pt")
    spidr.extract_spidr_discrete_units(dataset, tmp_path / "batched", "deu", "dev", "spidr.pt", batch_size=3)
    assert max(model.batch_sizes) == 3
    for layer in (3, 4):  # the layers with a codebook
        single = read_units(tmp_path / "single" / str(layer) / units_filename(GERMAN, "dev"))
        batched = read_units(tmp_path / "batched" / str(layer) / units_filename(GERMAN, "dev"))
        assert single == batched
        assert single == {fileid: [(t + layer) % N_CODES for t in range(n_frames(fileid))] for fileid in NUM_SAMPLES}


def test_spidr_continuous_features_with_batches(tmp_path: Path, dataset: Path, model: FakeModel) -> None:
    spidr.extract_spidr_continuous_features(dataset, tmp_path / "single", "deu", "dev", "spidr.pt", layers=2)
    spidr.extract_spidr_continuous_features(
        dataset, tmp_path / "batched", "deu", "dev", "spidr.pt", layers=2, batch_size=3
    )
    assert max(model.batch_sizes) == 3
    single, batched = read_features(tmp_path / "single"), read_features(tmp_path / "batched")
    assert single.keys() == batched.keys() == {f"2/deu/dev/{fileid}.pt" for fileid in NUM_SAMPLES}
    for name, frames in single.items():
        torch.testing.assert_close(batched[name], frames)
        assert batched[name].untyped_storage().nbytes() == frames.numel() * frames.element_size()  # not the batch


@pytest.mark.parametrize("batch_size", [1, 3])
def test_spidr_discrete_units_resume(tmp_path: Path, dataset: Path, model: FakeModel, batch_size: int) -> None:
    spidr.extract_spidr_discrete_units(dataset, tmp_path / "units", "deu", "dev", "spidr.pt")
    path = tmp_path / "units" / "4" / units_filename(GERMAN, "dev")
    expected = read_units(path)
    path.write_text("".join(path.read_text(encoding="utf-8").splitlines(keepends=True)[:2]), encoding="utf-8")
    model.batch_sizes.clear()
    spidr.extract_spidr_discrete_units(dataset, tmp_path / "units", "deu", "dev", "spidr.pt", batch_size=batch_size)
    assert sum(model.batch_sizes) == len(NUM_SAMPLES) - 2  # only the files missing from layer 4
    assert read_units(path) == expected
    model.batch_sizes.clear()
    spidr.extract_spidr_discrete_units(dataset, tmp_path / "units", "deu", "dev", "spidr.pt", batch_size=batch_size)
    assert model.batch_sizes == []  # nothing left to do


@pytest.mark.parametrize("architecture", ["hubert", "spidr"])
def test_continuous_features_resume(tmp_path: Path, dataset: Path, model: FakeModel, architecture: str) -> None:
    def extract() -> None:
        if architecture == "hubert":
            hubert.extract_hubert_continuous_features(
                dataset, tmp_path / "features", "deu", "dev", "it2.pt", layers=[1, 2]
            )
        else:
            spidr.extract_spidr_continuous_features(
                dataset, tmp_path / "features", "deu", "dev", "ckpt", layers=[1, 2]
            )

    extract()
    expected = read_features(tmp_path / "features")
    (tmp_path / "features" / "2" / "deu" / "dev" / "b.pt").unlink()
    (tmp_path / "features" / "1" / "deu" / "dev" / "d.pt").rename(
        tmp_path / "features" / "1" / "deu" / "dev" / "d.pt.part"
    )
    model.batch_sizes.clear()
    extract()
    assert model.batch_sizes == [1, 1]  # only the files b and d
    resumed = read_features(tmp_path / "features")
    assert resumed.keys() == expected.keys()
    for name, frames in expected.items():
        torch.testing.assert_close(resumed[name], frames)
    model.batch_sizes.clear()
    extract()
    assert model.batch_sizes == []


@pytest.mark.usefixtures("model")
def test_cli_extracts_spidr_units(tmp_path: Path, dataset: Path) -> None:
    common = [str(dataset), str(tmp_path / "units"), "spidr.pt", "--languages", "deu", "--splits", "dev"]
    extract_cli(["spidr", "units", *common, "--batch-size", "3"])
    for layer in (3, 4):
        units = read_units(tmp_path / "units" / str(layer) / units_filename(GERMAN, "dev"))
        assert units == {fileid: [(t + layer) % N_CODES for t in range(n_frames(fileid))] for fileid in NUM_SAMPLES}


@pytest.mark.usefixtures("model")
def test_cli_extracts_hubert_units_with_kmeans(tmp_path: Path, dataset: Path) -> None:
    joblib.dump(FakeKMeans(), tmp_path / "kmeans.joblib")
    common = [str(dataset), str(tmp_path / "units"), "it2.pt", "--languages", "deu", "--splits", "dev"]
    extract_cli(["hubert", "units", *common, "--kmeans", f"2={tmp_path / 'kmeans.joblib'}"])
    assert [p.parent.name for p in (tmp_path / "units").rglob("*.jsonl")] == ["2"]
    units = read_units(tmp_path / "units" / "2" / units_filename(GERMAN, "dev"))
    assert units == {fileid: [t + 2 for t in range(n_frames(fileid))] for fileid in NUM_SAMPLES}


def test_cli_extracts_all_languages_and_splits_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    called = MagicMock()
    monkeypatch.setattr(extract, "extract_spidr_continuous_features", called)
    extract_cli(["spidr", "features", "data", "features", "spidr.pt", "--layers", "5", "6", "--batch-size", "8"])
    assert [c.args[2:4] for c in called.call_args_list] == [
        (language.iso_639_3, split) for language in all_languages() for split in ("dev", "test")
    ]
    assert all(c.kwargs == {"layers": [5, 6], "batch_size": 8} for c in called.call_args_list)


@pytest.mark.parametrize(
    ("args", "match"),
    [
        (["hubert", "units"], "require at least one"),
        (["spidr", "units", "--kmeans", "6=kmeans.joblib"], "only applies to HuBERT units"),
        (["hubert", "features", "--batch-size", "2"], "must be 1"),
        (["hubert", "units", "--kmeans", "six=kmeans.joblib"], "invalid layer_and_path value"),
        (["hubert", "units", "--kmeans", "6=kmeans.joblib", "--layers", "6", "7"], "No `--kmeans` for the layers [7]"),
    ],
)
def test_cli_rejects_invalid_arguments(capsys: pytest.CaptureFixture[str], args: list[str], match: str) -> None:
    with pytest.raises(SystemExit):
        extract_cli([*args[:2], "data", "output", "ckpt", *args[2:]])
    assert match in capsys.readouterr().err
