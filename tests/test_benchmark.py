"""Tests for the benchmark orchestration (no dataset download required)."""

import json
from pathlib import Path
from typing import Literal

import polars as pl
import pytest

from discophon.benchmark import (
    available_languages_and_splits_for_units,
    benchmark_abx_continuous,
    benchmark_abx_discrete,
    benchmark_discovery,
)
from discophon.data import alignment_filename, item_filename, read_gold_annotations, units_filename
from discophon.evaluate import phoneme_discovery
from discophon.languages import get_language

from .test_validate import build_valid_dataset

EMPTY_FRAME_COLUMNS = ["language", "split", "metric", "score"]
SPEAKERS = {"f1": "s1", "f2": "s1", "f3": "s2", "f4": "s2"}


def write_synthetic_split(dataset: Path, predictions: Path, *, swapped_speaker: str | None = None) -> None:
    """Gold phones and predictions for German dev: each file alternates the phones `a` and `b` (100 ms each).

    The units (50 Hz) and the continuous features (one-hot units) match the phones perfectly, except for the files of
    `swapped_speaker`, where the two units are swapped.
    """
    torch = pytest.importorskip("torch")
    language = get_language("deu")
    alignment, items, units = (
        ["#file onset offset #phone"],
        ["#file onset offset #phone prev-phone next-phone speaker"],
        [],
    )
    (predictions / "deu" / "dev").mkdir(parents=True)
    for fileid, speaker in SPEAKERS.items():
        sequence = []
        for i, phone in enumerate("abababab"):
            alignment.append(f"{fileid} {i / 10:.1f} {(i + 1) / 10:.1f} {phone}")
            items.append(f"{fileid} {i / 10:.1f} {(i + 1) / 10:.1f} {phone} x y {speaker}")
            sequence += [int((phone == "a") == (speaker == swapped_speaker))] * 5
        units.append(json.dumps({"file": fileid, "units": sequence}))
        torch.save(
            torch.nn.functional.one_hot(torch.tensor(sequence), 2).float(),
            predictions / "deu" / "dev" / f"{fileid}.pt",
        )
    (dataset / "alignment" / alignment_filename(language, "dev")).write_text(
        "\n".join(alignment) + "\n", encoding="utf-8"
    )
    for kind in ["triphone", "phoneme"]:
        (dataset / "item" / item_filename(language, "dev", kind=kind)).write_text(
            "\n".join(items) + "\n", encoding="utf-8"
        )
    (predictions / units_filename(language, "dev")).write_text("\n".join(units) + "\n", encoding="utf-8")


def test_available_languages_and_splits_for_units(tmp_path: Path) -> None:
    (tmp_path / units_filename(get_language("deu"), "dev")).touch()
    (tmp_path / units_filename(get_language("eng"), "train-10h")).touch()
    found = available_languages_and_splits_for_units(tmp_path)
    assert (get_language("deu"), "dev") in found
    assert (get_language("eng"), "train-10h") in found


def test_available_languages_and_splits_for_units_skips_unknown_languages(tmp_path: Path) -> None:
    (tmp_path / units_filename(get_language("deu"), "dev")).touch()
    (tmp_path / "units-old-dev.jsonl").touch()
    assert available_languages_and_splits_for_units(tmp_path) == [(get_language("deu"), "dev")]


def test_benchmark_discovery_returns_empty_frame_when_no_units(tmp_path: Path) -> None:
    # valid dataset but no predicted units: the result is an empty, well-typed frame (not a crash)
    dataset = build_valid_dataset(tmp_path / "dataset")
    units = tmp_path / "units"
    units.mkdir()
    out = benchmark_discovery(dataset, units, kind="many-to-one")
    assert out.columns == EMPTY_FRAME_COLUMNS
    assert out.schema["score"] == pl.Float64
    assert out.is_empty()


def test_benchmark_abx_discrete_returns_empty_frame_when_no_units(tmp_path: Path) -> None:
    pytest.importorskip("fastabx")  # benchmark_abx_discrete imports discophon.abx, which needs the [abx] extra
    dataset = build_valid_dataset(tmp_path / "dataset")
    units = tmp_path / "units"
    units.mkdir()
    out = benchmark_abx_discrete(dataset, units)
    assert out.columns == EMPTY_FRAME_COLUMNS
    assert out.schema["score"] == pl.Float64
    assert out.is_empty()


def test_benchmark_abx_continuous_returns_empty_frame_when_no_features(tmp_path: Path) -> None:
    pytest.importorskip("fastabx")  # benchmark_abx_continuous imports discophon.abx, which needs the [abx] extra
    dataset = build_valid_dataset(tmp_path / "dataset")
    features = tmp_path / "features"
    features.mkdir()
    out = benchmark_abx_continuous(dataset, features)
    assert out.columns == EMPTY_FRAME_COLUMNS
    assert out.schema["score"] == pl.Float64
    assert out.is_empty()


@pytest.mark.parametrize(("kind", "n_units"), [("many-to-one", 256), ("one-to-one", 42)])
def test_benchmark_discovery_matches_direct_evaluation(
    tmp_path: Path, kind: Literal["many-to-one", "one-to-one"], n_units: int
) -> None:
    dataset, predictions = build_valid_dataset(tmp_path / "dataset"), tmp_path / "predictions"
    write_synthetic_split(dataset, predictions)
    out = benchmark_discovery(dataset, predictions, kind=kind)
    phones = read_gold_annotations(dataset / "alignment" / "alignment-deu-dev.txt")
    units = {fileid: [int(phone == "a") for phone in seq[::2]] for fileid, seq in phones.items()}
    expected = phoneme_discovery(units, phones, kind=kind, n_units=n_units, language="deu")
    assert out.columns == EMPTY_FRAME_COLUMNS
    assert dict(zip(out["metric"], out["score"], strict=True)) == expected
    assert set(out["language"]) == {"deu"}
    assert set(out["split"]) == {"dev"}
    assert expected["per"] == 0


def test_benchmark_abx_discrete_and_continuous(tmp_path: Path) -> None:
    pytest.importorskip("fastabx")
    dataset, predictions = build_valid_dataset(tmp_path / "dataset"), tmp_path / "predictions"
    write_synthetic_split(dataset, predictions, swapped_speaker="s2")
    discrete = benchmark_abx_discrete(dataset, predictions)
    continuous = benchmark_abx_continuous(dataset, predictions, kind="phoneme")
    assert dict(zip(discrete["metric"], discrete["score"], strict=True)) == {
        "triphone_abx_discrete_within_speaker": 0.0,
        "triphone_abx_discrete_across_speaker": 1.0,
    }
    assert set(continuous["metric"]) == {
        f"phoneme_abx_continuous_{speaker}_speaker_{context}_context"
        for speaker in ["within", "across"]
        for context in ["within", "any"]
    }
    assert set(continuous["language"]) == {"deu"}
