import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from discophon.baselines import spidr
from discophon.baselines.utils import link_best_checkpoint, read_completed_fileids


def test_read_completed_fileids_supports_resuming_and_deduplicates(tmp_path: Path) -> None:
    output = tmp_path / "units.jsonl"
    entries = [{"file": "a", "units": [1]}, {"file": "a", "units": [1]}, {"file": "b", "units": [2]}]
    output.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    assert read_completed_fileids(output) == {"a", "b"}


def test_read_completed_fileids_accepts_missing_output(tmp_path: Path) -> None:
    assert read_completed_fileids(tmp_path / "missing.jsonl") == set()


@pytest.mark.parametrize("bad", ['{"file":', '{"file": "b"}', '{"file": "b", "units": [null]}', "\n"])
def test_corrupt_resume_output_raises_without_modifying_file(tmp_path: Path, bad: str) -> None:
    output = tmp_path / "units.jsonl"
    contents = '{"file": "a", "units": [1]}\n' + bad
    output.write_text(contents, encoding="utf-8")
    with pytest.raises(ValueError, match=r"Corrupt units file .*line 2"):
        read_completed_fileids(output)
    assert output.read_text(encoding="utf-8") == contents


def test_best_checkpoint_link_can_be_updated(tmp_path: Path) -> None:
    link_best_checkpoint(tmp_path, "step_1000.pt")
    link_best_checkpoint(tmp_path, "final.pt")
    assert (tmp_path / "best.pt").readlink() == Path("final.pt")


def test_best_checkpoint_link_preserves_regular_file(tmp_path: Path) -> None:
    checkpoint = tmp_path / "best.pt"
    contents = b"checkpoint"
    checkpoint.write_bytes(contents)
    with pytest.raises(FileExistsError):
        link_best_checkpoint(tmp_path, "final.pt")
    assert checkpoint.read_bytes() == contents


def test_spidr_validation_reruns_ignore_aliases_and_old_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in ("set_seed", "setup_pytorch", "setup_environment", "patch_manifest_with_paths", "build_dataloader"):
        monkeypatch.setattr(spidr, name, MagicMock())
    monkeypatch.setattr(spidr.torch.cuda, "get_device_capability", lambda: (8, 0))
    build = MagicMock()
    monkeypatch.setattr(spidr, "build_model", build)
    monkeypatch.setattr(spidr, "validate_spidr", MagicMock(side_effect=[{"loss": 2.0}, {"loss": 1.0}] * 2))
    for filename in ("step_1000.pt", "step_2000.pt", "final.pt"):
        (tmp_path / filename).touch()
    link_best_checkpoint(tmp_path, "final.pt")
    output = tmp_path / "scores.jsonl"
    output.write_text(
        '{"step": 9999, "group": "other", "loss": 0.0}\n{"step": 1000, "group": "deu-dev", "loss": 0.0}\n',
        encoding="utf-8",
    )
    for _ in range(2):
        spidr.validate_all_spidr_checkpoints(output, tmp_path, tmp_path / "manifest-deu-dev.csv")
        assert (tmp_path / "best.pt").readlink() == Path("step_2000.pt")
    assert [call.kwargs["checkpoint"].name for call in build.call_args_list] == [
        "step_1000.pt",
        "step_2000.pt",
        "step_1000.pt",
        "step_2000.pt",
    ]
