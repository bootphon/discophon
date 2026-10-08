"""Smoke tests for the command-line entry points (argument wiring, end-to-end I/O)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from discophon.baselines import __main__ as baselines_main
from discophon.baselines.__main__ import cli as baselines_cli
from discophon.benchmark import cli as benchmark_cli
from discophon.evaluate import __main__ as evaluate_main
from discophon.evaluate.__main__ import cli as evaluate_cli

from .test_validate import build_valid_dataset


def _write_tiny_prediction(tmp_path: Path) -> tuple[Path, Path]:
    """2-phone alignment (10 ms phones) and matching units (20 ms step), perfectly aligned."""
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


@pytest.mark.parametrize(
    ("kind", "args", "n_units"),
    [("many-to-one", ["--n-phonemes", "2"], 256), ("one-to-one", ["--language", "deu"], 42)],
)
def test_evaluate_cli_infers_number_of_units(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str, args: list[str], n_units: int
) -> None:
    units, alignment = _write_tiny_prediction(tmp_path)
    called = MagicMock(return_value={})
    monkeypatch.setattr(evaluate_main, "phoneme_discovery", called)
    evaluate_cli([str(units), str(alignment), "--kind", kind, *args])
    assert called.call_args.kwargs["n_units"] == n_units
    assert called.call_args.kwargs["step_units"] == 20


def test_baselines_cli_dispatches_finetuning(monkeypatch: pytest.MonkeyPatch) -> None:
    hubert, spidr = MagicMock(), MagicMock()
    monkeypatch.setattr(baselines_main, "finetune_hubert", hubert)
    monkeypatch.setattr(baselines_main, "finetune_spidr", spidr)
    common = ["name", "project", "workdir", "ckpt.pt", "manifest.csv"]
    baselines_cli(["spidr", *common])
    spidr.assert_called_once_with("name", "project", Path("workdir"), Path("ckpt.pt"), "manifest.csv")
    baselines_cli(["hubert", *common, "--n-clusters", "500", "--layer", "9"])
    hubert.assert_called_once_with(
        "name", "project", Path("workdir"), Path("ckpt.pt"), "manifest.csv", n_clusters=500, target_layer=9
    )
    with pytest.raises(SystemExit):
        baselines_cli(["hubert", *common, "--layer", "9"])
    assert hubert.call_count == 1


def test_benchmark_cli_writes_output_file(tmp_path: Path) -> None:
    dataset = build_valid_dataset(tmp_path / "dataset")
    units = tmp_path / "units"
    units.mkdir()
    output = tmp_path / "scores.jsonl"
    benchmark_cli([str(dataset), str(units), str(output), "--benchmark", "discovery"])
    # no units available, so the run produces a valid (empty) output file rather than crashing
    assert output.exists()
    assert output.stat().st_size == 0
