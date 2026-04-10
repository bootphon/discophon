import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import joblib
import orjson
import torch
from filelock import FileLock
from minimal_hubert.config import hubert_data_config
from minimal_hubert.data import build_dataloader_with_labels
from minimal_hubert.model import HuBERTPretrain
from spidr.config import MaskingConfig
from spidr.environment import set_seed, setup_environment, setup_pytorch
from spidr.tools import init_logger
from torch.utils.data import DataLoader

from .finetune_hubert import patch_manifest_with_paths, patch_manifest_with_units

logger = logging.getLogger()


@torch.no_grad()
def validate(
    model: HuBERTPretrain,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_feature_loss = torch.zeros(1, device=device)
    mixed_precision = dtype != torch.float32
    for waveforms, labels, attn_mask, mask in loader:
        with torch.autocast("cuda", dtype, mixed_precision):
            loss, outputs = model(
                waveforms.to(device),
                labels.to(device),
                mask=mask.to(device),
                attention_mask=attn_mask.to(device),
            )
        total_loss += loss.mean()
        total_feature_loss += outputs["feature_loss"]
    total_loss /= len(loader)
    total_feature_loss /= len(loader)
    return {"loss": total_loss.item(), "feature_loss": total_feature_loss.item()}


def validate_all_checkpoints(
    manifest: str,
    checkpoints: str | Path,
    output: str | Path,
    *,
    seed: int = 0,
) -> None:
    init_logger()
    set_seed(seed)
    setup_pytorch(use_deterministic=False)
    setup_environment()
    device, dtype = torch.device("cuda"), torch.bfloat16
    loader = build_dataloader_with_labels(hubert_data_config(manifest), MaskingConfig())
    lock = FileLock(f"{output}.lock")
    for path in sorted(Path(checkpoints).glob("*.pt")):
        model = HuBERTPretrain.from_pretrained(path).to(device)
        losses = validate(model, loader, device, dtype)
        with lock, Path(output).open("ab") as f:
            f.write(orjson.dumps({"name": path.stem} | losses, option=orjson.OPT_APPEND_NEWLINE))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run validation for all existing HuBERT checkpoints")
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoints", type=Path)
    parser.add_argument("features", type=Path)
    # parser.add_argument("kmeans", type=Path)
    args = parser.parse_args()
    new_manifest = args.checkpoints / f"{args.manifest.stem.replace('manifest-', 'units-')}.jsonl"
    # kmeans = joblib.load(args.kmeans)
    kmeans = joblib.load(args.checkpoints / "kmeans.joblib")
    with NamedTemporaryFile(suffix=".jsonl") as tmpfile:
        patch_manifest_with_paths(args.manifest, tmpfile.name)
        patch_manifest_with_units(tmpfile.name, new_manifest, args.features, kmeans)
    validate_all_checkpoints(new_manifest, args.checkpoints, args.output)
