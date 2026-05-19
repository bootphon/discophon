"""SpidR finetuning."""

from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import orjson
import torch
import wandb
from spidr.checkpoint import Checkpointer
from spidr.config import MaskingConfig
from spidr.data import build_dataloader
from spidr.environment import set_seed, setup_environment, setup_pytorch
from spidr.models import build_model
from spidr.tools import AverageMeters, profiler_context
from torch import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm

from discophon.baselines.utils import (
    LOG_INTERVAL,
    SAVE_INTERVAL,
    SEED,
    DiscophonAudioDataset,
    ft_optimizer_config,
    get_target_layers,
    patch_manifest_with_paths,
    spidr_ft_data_config,
    tristage_scheduler,
)


def finetune_spidr(  # noqa: PLR0914
    name: str,
    project: str,
    workdir: Path,
    checkpoint: Path,
    manifest: str,
) -> None:
    """Finetune SpidR on DiscoPhon data with the default configuration.

    Args:
        name: Run name
        project: Run project
        workdir: Working directory for checkpoints and Wandb logs
        checkpoint: Path to the pretrained checkpoint
        manifest: Path to the manifest
    """
    cfg = ft_optimizer_config()
    with ExitStack() as stack:
        # Common setup
        set_seed(SEED)
        setup_pytorch(use_deterministic=False)
        setup_environment()
        rundir = workdir / project / name
        rundir.mkdir(parents=True, exist_ok=True)
        wandb.init(project=project, name=name, mode="offline", dir=workdir)
        stack.callback(wandb.finish)
        device = torch.device("cuda")

        # SpidR data setup
        tempfile = stack.enter_context(NamedTemporaryFile(suffix=".csv"))
        patch_manifest_with_paths(manifest, tempfile.name)
        loader = build_dataloader(spidr_ft_data_config(tempfile.name), MaskingConfig())

        # SpidR
        model = build_model(model_type="spidr", checkpoint=checkpoint).to(device).train()

        # Common setup
        optimizer = AdamW(
            model.parameters(),
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
            betas=cfg.betas,
            eps=cfg.eps,
            fused=True,
        )
        scaler = GradScaler("cuda")
        scheduler = tristage_scheduler(
            optimizer,
            warmup_steps=cfg.warmup_steps,
            hold_steps=cfg.hold_steps,
            decay_steps=cfg.decay_steps,
        )
        ckpt = Checkpointer(rundir, SAVE_INTERVAL)
        ckpt.init_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        step, epoch = int(ckpt.step), int(ckpt.epoch)
        stack.callback(lambda: ckpt.save(step, epoch))
        meters = AverageMeters(["loss", "grad_norm", "batch_size", "target_ppl", "pred_ppl"], device=device)
        profiler = stack.enter_context(profiler_context(rundir / "trace.html"))
        pbar = stack.enter_context(tqdm(total=cfg.max_steps, initial=step))
        if torch.cuda.get_device_capability() >= (8, 0):
            model.compile(dynamic=True)
            dtype = torch.bfloat16
        else:
            dtype = torch.float16

        # Training loop
        while step < cfg.max_steps:
            epoch += 1
            loader.batch_sampler.set_epoch(epoch)  # ty: ignore[unresolved-attribute]
            for waveforms, attn_mask, mask in loader:
                if step >= cfg.max_steps:
                    break
                with torch.autocast("cuda", dtype):
                    loss, outputs = model(
                        waveforms.to(device),
                        mask=mask.to(device),
                        attention_mask=attn_mask.to(device),
                    )
                loss = loss.mean()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                grad_norm = clip_grad_norm_(model.parameters(), cfg.max_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                lr = scheduler.get_last_lr()[0]
                scheduler.step()
                step += 1
                meters.update(
                    loss=loss.detach(),
                    batch_size=waveforms.size(0),
                    grad_norm=grad_norm,
                    target_ppl=outputs["target_ppl"],
                    pred_ppl=outputs["pred_ppl"],
                )
                pbar.update()
                if step % LOG_INTERVAL == 0:
                    infos = meters.pop() | {"lr": lr, "step": step, "epoch": epoch}
                    wandb.log(infos)
                    pbar.set_postfix(loss=infos["loss"], target_ppl=infos["target_ppl"], pred_ppl=infos["pred_ppl"])
                ckpt.save(step, epoch)
                profiler.step()
        ckpt.save_final(step, epoch)


@torch.inference_mode()
def extract_spidr_discrete_units(
    path_dataset: str | Path,
    path_units: str | Path,
    language: str,
    split: Literal["dev", "test", "train-10min", "train-1h", "train-10h"],
    checkpoint: str | Path,
    *,
    layers: int | Iterable[int] | None = None,
) -> None:
    """Extract SpidR discrete units for all utterances of a DiscoPhon split.

    The units are taken from the codebooks of the student network (one codebook per quantized layer).
    For each requested layer, the units are written to a JSONL file at
    `path_units / {layer} / units-{iso_639_3}-{split}.jsonl`, with one entry per
    utterance with keys `file` ([`str`][]) and `units` (`list[int]`).

    Args:
        path_dataset: Path to the DiscoPhon dataset.
        path_units: Output path used as a template. Its parent directory and filename stem
            determine where the per-layer JSONL files are written.
        language: Language identifier resolved by [`get_language`][discophon.languages.get_language],
            either name or ISO 639-3 code.
        split: Dataset split to process.
        checkpoint: Path to the SpidR checkpoint.
        layers: Layers to extract. If `None`, all layers with a codebook are used.
    """
    path_units = Path(path_units)
    dataset = DiscophonAudioDataset(path_dataset, language, split, normalize=True)
    model = build_model(model_type="spidr", checkpoint=checkpoint).eval().cuda()
    available = [len(model.student.layers) - model.num_codebooks + i + 1 for i in range(len(model.student.layers))]
    layers = get_target_layers(layers, available)
    for fileid, waveform in tqdm(dataset, desc=f"{dataset.language.iso_639_3}-{dataset.split}"):
        all_features = model.get_codebooks(waveform.unsqueeze(0).cuda())
        for layer, features in enumerate(all_features):
            if features is None or layer + 1 not in layers:
                continue
            units = features.squeeze().argmax(dim=-1).cpu().numpy().tolist()
            entry = {"file": fileid, "units": units}
            jsonl = path_units / f"{layer + 1}" / f"units-{dataset.language.iso_639_3}-{dataset.split}.jsonl"
            jsonl.parent.mkdir(exist_ok=True, parents=True)
            with jsonl.open("ab") as f:
                f.write(orjson.dumps(entry, option=orjson.OPT_APPEND_NEWLINE))


@torch.inference_mode()
def extract_spidr_continuous_features(
    path_dataset: str | Path,
    path_features: str | Path,
    language: str,
    split: Literal["dev", "test", "train-10min", "train-1h", "train-10h"],
    checkpoint: str | Path,
    *,
    layers: int | Iterable[int] | None = None,
) -> None:
    """Extract SpidR continuous features for all utterances of a DiscoPhon split.

    For each requested layer, the features are saved as PyTorch tensors at
    `path_features / {layer} / {iso_639_3} / {split} / {fileid}.pt`.

    Args:
        path_dataset: Path to the DiscoPhon dataset.
        path_features: Output directory under which per-layer feature tensors are written.
        language: Language identifier resolved by [`get_language`][discophon.languages.get_language],
            either name or ISO 639-3 code.
        split: Dataset split to process.
        checkpoint: Path to the SpidR checkpoint.
        layers: Layers to extract. If `None`, all student layers are used.
    """
    path_features = Path(path_features)
    dataset = DiscophonAudioDataset(path_dataset, language, split, normalize=True)
    model = build_model(model_type="spidr", checkpoint=checkpoint).eval().cuda()
    layers = get_target_layers(layers, [i + 1 for i in range(len(model.student.layers))])
    for fileid, waveform in tqdm(dataset, desc=f"{dataset.language.iso_639_3}-{dataset.split}"):
        all_features = model.get_intermediate_outputs(waveform.unsqueeze(0).cuda())
        for layer, features in enumerate(all_features):
            if layer + 1 not in layers:
                continue
            path = path_features / f"{layer + 1}" / dataset.language.iso_639_3 / dataset.split / f"{fileid}.pt"
            path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(features.squeeze().cpu(), path)
