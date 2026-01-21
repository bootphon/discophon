import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch
from spidr.config import MaskingConfig
from spidr.data import build_dataloader
from spidr.environment import set_seed, setup_environment, setup_pytorch
from spidr.models import DinoSR, build_model
from spidr.tools import init_logger
from torch.utils.data import DataLoader
from tqdm import tqdm

from .finetune_spidr import patch_manifest_with_paths, spidr_ft_data_config

logger = logging.getLogger()


@torch.no_grad()
def validate(
    model: DinoSR,
    loader: DataLoader,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, float]:
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_pred_ppl = torch.zeros(1, device=device)
    total_target_ppl = torch.zeros(1, device=device)
    for waveforms, attn_mask, mask in loader:
        with torch.autocast("cuda", dtype):
            loss, outputs = model(
                waveforms.to(device),
                mask=mask.to(device),
                attention_mask=attn_mask.to(device),
            )
        total_loss += loss.mean()
        total_target_ppl += outputs["target_ppl"]
        total_pred_ppl += outputs["pred_ppl"]
    total_loss /= len(loader)
    total_target_ppl /= len(loader)
    total_pred_ppl /= len(loader)
    return {"loss": total_loss.item(), "target_ppl": total_target_ppl.item(), "pred_ppl": total_pred_ppl.item()}


def validate_all_checkpoints(
    output: str | Path,
    manifest: str,
    checkpoints: str | Path,
    *,
    seed: int = 0,
) -> None:
    init_logger()
    set_seed(seed)
    setup_pytorch(use_deterministic=False)
    setup_environment()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16
    loader = build_dataloader(spidr_ft_data_config(manifest), MaskingConfig())
    paths = sorted(Path(checkpoints).glob("*.pt"))
    for path in tqdm(paths):
        model = build_model(model_type="spidr", checkpoint=path).to(device)
        losses = validate(model, loader, device, dtype)
        with Path(output).open("a") as f:
            f.write(json.dumps({"name": path.stem} | losses) + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("checkpoints", type=Path)
    args = parser.parse_args()
    with NamedTemporaryFile(suffix=".csv") as tmpfile:
        patch_manifest_with_paths(args.manifest, tmpfile.name)
        validate_all_checkpoints(args.output, tmpfile.name, args.checkpoints)
