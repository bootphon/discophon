"""SpidR finetuning."""

from collections.abc import Iterable
from contextlib import ExitStack
from dataclasses import replace
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Literal

import orjson
import polars as pl
import torch
import wandb
from spidr.checkpoint import Checkpointer
from spidr.config import MaskingConfig
from spidr.data import build_dataloader
from spidr.environment import set_seed, setup_environment, setup_pytorch
from spidr.models import DinoSR, build_model
from spidr.tools import AverageMeters, profiler_context
from torch import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from discophon.baselines.utils import (
    LOG_INTERVAL,
    SAVE_INTERVAL,
    SEED,
    DiscophonAudioDataset,
    build_inference_dataloader,
    ft_optimizer_config,
    get_target_layers,
    link_best_checkpoint,
    patch_manifest_with_paths,
    read_completed_fileids,
    spidr_ft_data_config,
    tristage_scheduler,
)
from discophon.data import units_filename


def finetune_spidr(  # ruff: ignore[too-many-locals, too-many-statements]
    name: str,
    project: str,
    workdir: str | Path,
    checkpoint: str | Path,
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
        rundir = Path(workdir) / project / name
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
        ckpt.load_existing_run()
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
            for waveforms, attn_mask, mask_indices in loader:
                if step >= cfg.max_steps:
                    break
                with torch.autocast("cuda", dtype):
                    loss, outputs = model(
                        waveforms.to(device),
                        mask_indices=mask_indices.to(device),
                        attention_mask=attn_mask.to(device) if attn_mask is not None else None,
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


@torch.no_grad()
def validate_spidr(model: DinoSR, loader: DataLoader, device: torch.device, dtype: torch.dtype) -> dict[str, float]:
    model.eval()
    total_loss = torch.zeros(1, device=device)
    total_pred_ppl = torch.zeros(1, device=device)
    total_target_ppl = torch.zeros(1, device=device)
    for waveforms, attn_mask, mask_indices in loader:
        with torch.autocast("cuda", dtype):
            loss, outputs = model(
                waveforms.to(device),
                mask_indices=mask_indices.to(device),
                attention_mask=attn_mask.to(device) if attn_mask is not None else None,
            )
        total_loss += loss.mean()
        total_target_ppl += outputs["target_ppl"]
        total_pred_ppl += outputs["pred_ppl"]
    total_loss /= len(loader)
    total_target_ppl /= len(loader)
    total_pred_ppl /= len(loader)
    return {"loss": total_loss.item(), "target_ppl": total_target_ppl.item(), "pred_ppl": total_pred_ppl.item()}


def validate_all_spidr_checkpoints(
    output: str | Path,
    checkpoints: str | Path,
    manifest: str | Path,
    *,
    seed: int = 0,
) -> None:
    """Compute the validation loss of every checkpoint of a SpidR finetuning, and link the best one to `best.pt`.

    Every checkpoint is evaluated with the same masks and crops.

    Args:
        output: JSONL file to which the losses of each checkpoint are appended
        checkpoints: Directory with the `step_*.pt` checkpoints, where `best.pt` is created
        manifest: Manifest of the validation files
        seed: Random seed of the masks and crops

    Raises:
        ValueError: If there is no checkpoint in `checkpoints`.

    """
    set_seed(seed)
    setup_pytorch(use_deterministic=False)
    setup_environment()
    device = torch.device("cuda")
    dtype = torch.bfloat16 if torch.cuda.get_device_capability() >= (8, 0) else torch.float16
    with NamedTemporaryFile(suffix=".csv") as tempfile:
        patch_manifest_with_paths(manifest, tempfile.name)
        cfg = replace(spidr_ft_data_config(tempfile.name), persistent_workers=False)
        loader = build_dataloader(cfg, MaskingConfig())
    paths = sorted(Path(checkpoints).glob("step_*.pt"))
    if not paths:
        raise ValueError(f"No step checkpoints found in {checkpoints}")
    group = Path(manifest).stem.removeprefix("manifest-")
    results = []
    for path in tqdm(paths):
        step = int(path.stem.removeprefix("step_"))
        model = build_model(model_type="spidr", checkpoint=path).to(device)
        # Same masks and crops for every checkpoint
        loader.generator.manual_seed(seed)  # ty: ignore[unresolved-attribute]
        losses = validate_spidr(model, loader, device, dtype)
        results.append({"step": step, "group": group} | losses)
        with Path(output).open("ab") as f:
            f.write(orjson.dumps({"step": step, "group": group} | losses, option=orjson.OPT_APPEND_NEWLINE))

    best_step = (
        pl.DataFrame(results).sort("step").filter(pl.col("loss") == pl.col("loss").min()).tail(1).to_dicts()[0]["step"]
    )
    link_best_checkpoint(Path(checkpoints), f"step_{best_step}.pt")


@torch.inference_mode()
def extract_spidr_discrete_units(
    path_dataset: str | Path,
    path_units: str | Path,
    language: str,
    split: Literal["dev", "test", "train-10min", "train-1h", "train-10h"],
    checkpoint: str | Path,
    *,
    layers: int | Iterable[int] | None = None,
    batch_size: int = 1,
) -> None:
    """Extract SpidR discrete units for all utterances of a DiscoPhon split.

    The units are taken from the codebooks of the student network (one codebook per quantized layer).
    For each requested layer, the units are written to a JSONL file at
    `path_units / {layer} / units-{iso_639_3}-{split}.jsonl`, with one entry per
    utterance with keys `file` ([`str`][]) and `units` (`list[int]`).

    Args:
        path_dataset: Path to the DiscoPhon dataset.
        path_units: Output directory under which the per-layer JSONL files are written.
        language: Language identifier resolved by [`get_language`][discophon.languages.get_language],
            either name or ISO 639-3 code.
        split: Dataset split to process.
        checkpoint: Path to the SpidR checkpoint.
        layers: Layers to extract. If `None`, all layers with a codebook are used.
        batch_size: Number of utterances per forward pass. Padded batches give the same units up to numerical
            precision, and a batch size of 1 reproduces the published units exactly.

    """
    path_units = Path(path_units)
    dataset = DiscophonAudioDataset(path_dataset, language, split, normalize=True)
    model = build_model(model_type="spidr", checkpoint=checkpoint).eval().cuda()
    available = [len(model.student.layers) - model.num_codebooks + i + 1 for i in range(model.num_codebooks)]
    layers = get_target_layers(layers, available)
    outputs = {layer: path_units / f"{layer}" / units_filename(dataset.language, dataset.split) for layer in layers}
    completed = {layer: read_completed_fileids(path) for layer, path in outputs.items()}
    pending = [any(fileid not in completed[layer] for layer in outputs) for fileid in dataset.manifest["fileid"]]
    dataset.manifest = dataset.manifest.filter(pl.Series(pending, dtype=pl.Boolean))
    if dataset.manifest.is_empty():
        return
    loader = build_inference_dataloader(dataset, batch_size)
    for fileids, waveforms, attn_mask, feat_lengths in tqdm(
        loader, desc=f"{dataset.language.iso_639_3}-{dataset.split}"
    ):
        mask = attn_mask.cuda() if attn_mask is not None else None
        all_features = model.get_codebooks(waveforms.cuda(), attention_mask=mask)
        for layer, features in enumerate(all_features, start=1):
            if features is None or layer not in outputs:
                continue
            for fileid, logits, length in zip(fileids, features, feat_lengths.tolist(), strict=True):
                if fileid in completed[layer]:
                    continue
                units = logits[:length].argmax(dim=-1).cpu().numpy().tolist()
                entry = {"file": fileid, "units": units}
                jsonl = outputs[layer]
                jsonl.parent.mkdir(exist_ok=True, parents=True)
                with jsonl.open("ab") as f:
                    f.write(orjson.dumps(entry, option=orjson.OPT_APPEND_NEWLINE))
                completed[layer].add(fileid)


@torch.inference_mode()
def extract_spidr_continuous_features(
    path_dataset: str | Path,
    path_features: str | Path,
    language: str,
    split: Literal["dev", "test", "train-10min", "train-1h", "train-10h"],
    checkpoint: str | Path,
    *,
    layers: int | Iterable[int] | None = None,
    batch_size: int = 1,
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
        batch_size: Number of utterances per forward pass. Padded batches give the same features up to numerical
            precision, and a batch size of 1 reproduces the published features exactly.

    Files whose features already exist for all requested layers are skipped, so the extraction can be resumed.

    """
    path_features = Path(path_features)
    dataset = DiscophonAudioDataset(path_dataset, language, split, normalize=True)
    model = build_model(model_type="spidr", checkpoint=checkpoint).eval().cuda()
    layers = get_target_layers(layers, [i + 1 for i in range(len(model.student.layers))])
    directories = {layer: path_features / f"{layer}" / dataset.language.iso_639_3 / dataset.split for layer in layers}
    pending = [
        any(not (directory / f"{fileid}.pt").is_file() for directory in directories.values())
        for fileid in dataset.manifest["fileid"]
    ]
    dataset.manifest = dataset.manifest.filter(pl.Series(pending, dtype=pl.Boolean))
    if dataset.manifest.is_empty():
        return
    loader = build_inference_dataloader(dataset, batch_size)
    for fileids, waveforms, attn_mask, feat_lengths in tqdm(
        loader, desc=f"{dataset.language.iso_639_3}-{dataset.split}"
    ):
        mask = attn_mask.cuda() if attn_mask is not None else None
        all_features = model.get_intermediate_outputs(waveforms.cuda(), attention_mask=mask)
        for layer, features in enumerate(all_features):
            if layer + 1 not in layers:
                continue
            for fileid, frames, length in zip(fileids, features, feat_lengths.tolist(), strict=True):
                path = directories[layer + 1] / f"{fileid}.pt"
                path.parent.mkdir(exist_ok=True, parents=True)
                partial = path.with_suffix(".pt.part")  # Written atomically to resume safely if interrupted
                torch.save(frames[:length].clone().cpu(), partial)  # Clone to not save the storage of the batch
                partial.replace(path)
