"""Smoke tests for the command-line entry points (argument wiring, end-to-end I/O)."""

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from discophon.baselines import __main__ as baselines_main
from discophon.baselines.__main__ import cli as baselines_cli
from discophon.benchmark import NoPredictionsError
from discophon.benchmark import cli as benchmark_cli
from discophon.evaluate import __main__ as evaluate_main
from discophon.evaluate.__main__ import cli as evaluate_cli

from .test_benchmark import write_synthetic_split
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


def test_benchmark_cli_appends_scores_to_output_file(tmp_path: Path) -> None:
    dataset, units = build_valid_dataset(tmp_path / "dataset"), tmp_path / "units"
    write_synthetic_split(dataset, units)
    output = tmp_path / "scores.jsonl"
    for _ in range(2):
        benchmark_cli([str(dataset), str(units), str(output), "--benchmark", "discovery"])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 2 * 4  # appended: 4 metrics for German dev, twice
    assert {(row["language"], row["split"]) for row in rows} == {("deu", "dev")}


def test_benchmark_cli_fails_without_units(tmp_path: Path) -> None:
    dataset = build_valid_dataset(tmp_path / "dataset")
    units = tmp_path / "units"
    units.mkdir()
    output = tmp_path / "scores.jsonl"
    with pytest.raises(NoPredictionsError):
        benchmark_cli([str(dataset), str(units), str(output), "--benchmark", "discovery"])
    assert not output.exists()


@pytest.mark.parametrize(
    ("args", "match"),
    [
        ([], "one of the arguments --language --n-phonemes is required"),
        (["--language", "deu", "--n-phonemes", "41"], "not allowed with argument --language"),
        (["--language", "klingon"], "invalid get_language value: 'klingon'"),
    ],
)
def test_evaluate_cli_rejects_invalid_targets(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], args: list[str], match: str
) -> None:
    units, alignment = _write_tiny_prediction(tmp_path)
    with pytest.raises(SystemExit):
        evaluate_cli([str(units), str(alignment), *args])
    assert match in capsys.readouterr().err


def test_abx_cli_rejects_unknown_inputs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    abx_cli = pytest.importorskip("discophon.abx").cli
    with pytest.raises(SystemExit):
        abx_cli([str(tmp_path / "triphone.item"), str(tmp_path / "units.txt"), "--frequency", "50"])
    assert "Expected a directory of features or a .jsonl units file" in capsys.readouterr().err


def parse_abx_output(output: str) -> dict[str, float]:
    return {key: float(score.removesuffix("%")) for key, score in (line.split(":\t") for line in output.splitlines())}


def test_abx_cli_uses_units_files_and_feature_directories(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    abx_cli = pytest.importorskip("discophon.abx").cli
    dataset, predictions = build_valid_dataset(tmp_path / "dataset"), tmp_path / "predictions"
    write_synthetic_split(dataset, predictions, swapped_speaker="s2")  # perfect within speakers, swapped across
    abx_cli(
        [
            str(dataset / "item" / "phoneme-deu-dev.item"),
            str(predictions / "units-deu-dev.jsonl"),
            "--frequency",
            "50",
            "--kind",
            "phoneme",
        ]
    )
    assert parse_abx_output(capsys.readouterr().out) == {
        "within_speaker_within_context": 0.0,
        "across_speaker_within_context": 100.0,
        "within_speaker_any_context": 0.0,
        "across_speaker_any_context": 100.0,
    }
    abx_cli([str(dataset / "item" / "triphone-deu-dev.item"), str(predictions / "deu" / "dev"), "--frequency", "50"])
    assert parse_abx_output(capsys.readouterr().out) == {"within_speaker": 0.0, "across_speaker": 100.0}


@pytest.mark.parametrize(
    ("args", "expected"),
    [
        (
            ["--benchmark", "abx-discrete", "--abx-kind", "phoneme"],
            {
                "phoneme_abx_discrete_within_speaker_within_context": 0.0,
                "phoneme_abx_discrete_across_speaker_within_context": 1.0,
                "phoneme_abx_discrete_within_speaker_any_context": 0.0,
                "phoneme_abx_discrete_across_speaker_any_context": 1.0,
            },
        ),
        (
            ["--benchmark", "abx-continuous"],
            {"triphone_abx_continuous_within_speaker": 0.0, "triphone_abx_continuous_across_speaker": 1.0},
        ),
    ],
)
def test_benchmark_cli_abx(tmp_path: Path, args: list[str], expected: dict[str, float]) -> None:
    pytest.importorskip("fastabx")
    dataset, predictions = build_valid_dataset(tmp_path / "dataset"), tmp_path / "predictions"
    write_synthetic_split(dataset, predictions, swapped_speaker="s2")
    output = tmp_path / "scores.jsonl"
    benchmark_cli([str(dataset), str(predictions), str(output), *args])
    rows = [json.loads(line) for line in output.read_text(encoding="utf-8").splitlines()]
    assert {(row["language"], row["split"]) for row in rows} == {("deu", "dev")}
    assert {row["metric"]: row["score"] for row in rows} == expected
