"""Smoke tests for the command-line entry points (argument wiring, end-to-end I/O)."""

import json
from pathlib import Path

import pytest

from discophon.benchmark import cli as benchmark_cli
from discophon.evaluate.__main__ import cli as evaluate_cli

from .test_validate import build_valid_dataset


def _write_tiny_prediction(tmp_path: Path) -> tuple[Path, Path]:
    """A 2-phone alignment (10 ms phones) and matching units (20 ms step), perfectly aligned."""
    alignment = tmp_path / "alignment.txt"
    alignment.write_text("#file onset offset #phone\nf 0 0.02 a\nf 0.02 0.04 b\n", encoding="utf-8")
    units = tmp_path / "units.jsonl"
    units.write_text('{"file": "f", "units": [0, 1]}\n', encoding="utf-8")
    return units, alignment


def test_evaluate_cli_prints_discovery_metrics(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    units, alignment = _write_tiny_prediction(tmp_path)
    evaluate_cli([str(units), str(alignment), "--n-units", "2", "--n-phonemes", "2"])
    scores = json.loads(capsys.readouterr().out)
    assert set(scores) == {"pnmi", "per", "f1", "r_val"}
    assert all(isinstance(v, float) for v in scores.values())


def test_benchmark_cli_writes_output_file(tmp_path: Path) -> None:
    dataset = build_valid_dataset(tmp_path / "dataset")
    units = tmp_path / "units"
    units.mkdir()
    output = tmp_path / "scores.jsonl"
    benchmark_cli([str(dataset), str(units), str(output), "--benchmark", "discovery"])
    # no units available, so the run produces a valid (empty) output file rather than crashing
    assert output.exists()
    assert output.stat().st_size == 0
