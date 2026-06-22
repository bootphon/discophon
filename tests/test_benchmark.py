"""Tests for the benchmark orchestration (no dataset download required)."""

from pathlib import Path

import polars as pl
import pytest

from discophon.benchmark import (
    available_languages_and_splits_for_units,
    benchmark_abx_continuous,
    benchmark_abx_discrete,
    benchmark_discovery,
)
from discophon.data import units_filename
from discophon.languages import get_language

from .test_validate import build_valid_dataset

EMPTY_FRAME_COLUMNS = ["language", "split", "metric", "score"]


def test_available_languages_and_splits_for_units(tmp_path: Path) -> None:
    (tmp_path / units_filename(get_language("deu"), "dev")).touch()
    (tmp_path / units_filename(get_language("eng"), "train-10h")).touch()
    found = available_languages_and_splits_for_units(tmp_path)
    assert (get_language("deu"), "dev") in found
    assert (get_language("eng"), "train-10h") in found


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
