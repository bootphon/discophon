import json
from pathlib import Path
from unittest.mock import MagicMock, call

import pytest
import torch

from discophon.baselines import hubert, spidr
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
    for name in ("set_seed", "setup_pytorch", "setup_environment", "patch_manifest_with_paths"):
        monkeypatch.setattr(spidr, name, MagicMock())
    monkeypatch.setattr(spidr.torch.cuda, "get_device_capability", lambda: (8, 0))
    build = MagicMock()
    monkeypatch.setattr(spidr, "build_model", build)
    loader = MagicMock()
    monkeypatch.setattr(spidr, "build_dataloader", MagicMock(return_value=loader))
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
    assert loader.generator.manual_seed.call_args_list == [call(0)] * 4  # same masks for every checkpoint
    assert [call.kwargs["checkpoint"].name for call in build.call_args_list] == [
        "step_1000.pt",
        "step_2000.pt",
        "step_1000.pt",
        "step_2000.pt",
    ]


def test_hubert_validation_reruns_ignore_aliases_and_old_scores(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for name in (
        "set_seed",
        "setup_pytorch",
        "setup_environment",
        "patch_manifest_with_paths",
        "compute_and_save_hubert_features",
        "patch_manifest_with_units",
    ):
        monkeypatch.setattr(hubert, name, MagicMock())
    monkeypatch.setattr(hubert.joblib, "load", MagicMock())
    monkeypatch.setattr(hubert.torch.cuda, "get_device_capability", lambda: (8, 0))
    build = MagicMock()
    monkeypatch.setattr(hubert.HuBERTPretrain, "from_pretrained", build)
    loader = MagicMock()
    monkeypatch.setattr(hubert, "build_dataloader_with_labels", MagicMock(return_value=loader))
    monkeypatch.setattr(hubert, "validate_hubert", MagicMock(side_effect=[{"loss": 2.0}, {"loss": 1.0}] * 2))
    for filename in ("step_1000.pt", "step_2000.pt", "final.pt"):
        (tmp_path / filename).touch()
    link_best_checkpoint(tmp_path, "final.pt")
    output = tmp_path / "scores.jsonl"
    output.write_text(
        '{"step": 9999, "group": "other", "loss": 0.0}\n{"step": 1000, "group": "deu-dev", "loss": 0.0}\n',
        encoding="utf-8",
    )
    for _ in range(2):
        hubert.validate_all_hubert_checkpoints(output, tmp_path, tmp_path / "manifest-deu-dev.csv", "it2.pt", 11)
        assert (tmp_path / "best.pt").readlink() == Path("step_2000.pt")
    assert loader.generator.manual_seed.call_args_list == [call(0)] * 4  # same masks for every checkpoint
    assert [call.args[0].name for call in build.call_args_list] == [
        "step_1000.pt",
        "step_2000.pt",
        "step_1000.pt",
        "step_2000.pt",
    ]


def test_hubert_finetuning_resumes_with_the_same_targets(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in (
        "set_seed",
        "setup_pytorch",
        "setup_environment",
        "wandb",
        "joblib",
        "patch_manifest_with_paths",
        "AdamW",
        "GradScaler",
        "tristage_scheduler",
        "AverageMeters",
        "profiler_context",
        "tqdm",
    ):
        monkeypatch.setattr(hubert, name, MagicMock())
    fit_kmeans, build_loader, checkpointer, hubert_pretrain = MagicMock(), MagicMock(), MagicMock(), MagicMock()
    monkeypatch.setattr(hubert, "HuBERTPretrain", hubert_pretrain)
    monkeypatch.setattr(hubert, "fit_kmeans_from_checkpoint", fit_kmeans)
    monkeypatch.setattr(hubert, "build_dataloader_with_labels", build_loader)
    monkeypatch.setattr(hubert, "Checkpointer", checkpointer)
    monkeypatch.setattr(hubert.torch.cuda, "get_device_capability", lambda: (8, 0))
    pretrained = hubert_pretrain.from_pretrained.return_value
    pretrained.logit_generator.label_embeddings = torch.zeros(4, 3)  # pretrained on 4 targets, with final_dim 3
    checkpointer.return_value.step = hubert.ft_optimizer_config().max_steps  # skip the training loop
    checkpointer.return_value.epoch = 0
    patch = MagicMock(side_effect=lambda _src, dest, *_: Path(dest).write_text("{}", encoding="utf-8"))
    monkeypatch.setattr(hubert, "patch_manifest_with_units", patch)
    for workdir in (tmp_path, str(tmp_path)):
        hubert.finetune_hubert(
            "run", "project", workdir, tmp_path / "it2.pt", "manifest.csv", n_clusters=8, target_layer=6
        )
    manifest = tmp_path / "project" / "run" / "manifest-with-units.jsonl"
    assert fit_kmeans.call_count == 1
    assert patch.call_count == 1
    assert manifest.read_text(encoding="utf-8") == "{}"
    assert build_loader.call_count == 2
    assert all(c.args[0].manifest == str(manifest) for c in build_loader.call_args_list)
    hubert_pretrain.from_pretrained.assert_called_with(tmp_path / "it2.pt")
    # New label embeddings for the new targets, initialized uniformly in [0, 1)
    assert isinstance(pretrained.logit_generator.label_embeddings, torch.nn.Parameter)
    assert pretrained.logit_generator.label_embeddings.shape == (8, 3)
    assert (
        0 <= pretrained.logit_generator.label_embeddings.min() <= pretrained.logit_generator.label_embeddings.max() < 1
    )
    assert pretrained.num_classes == 8
