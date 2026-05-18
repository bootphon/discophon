"""HuBERT finetuning."""

import logging
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import joblib
import torch
import wandb
from minimal_hubert import HuBERTPretrain
from minimal_hubert.data import build_dataloader_with_labels
from minimal_hubert.features import compute_and_save_hubert_features
from minimal_hubert.kmeans import build_kmeans
from sklearn.cluster import MiniBatchKMeans
from spidr.checkpoint import Checkpointer
from spidr.config import MaskingConfig
from spidr.environment import set_seed, setup_environment, setup_pytorch
from spidr.tools import AverageMeters, profiler_context
from torch import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW
from tqdm import tqdm

from discophon.baselines.utils import (
    LOG_INTERVAL,
    SAVE_INTERVAL,
    SEED,
    ft_optimizer_config,
    hubert_ft_data_config,
    patch_manifest_with_paths,
    patch_manifest_with_units,
    tristage_scheduler,
)

logger = logging.getLogger()


def fit_kmeans_from_checkpoint(
    manifest: str,
    checkpoint: str | Path,
    root_features: str | Path,
    layer: int,
    n_clusters: int,
    seed: int,
) -> MiniBatchKMeans:
    compute_and_save_hubert_features(manifest, root_features, checkpoint, layer)
    kmeans = build_kmeans(n_clusters, seed=seed)
    features = torch.concat([torch.load(p) for p in Path(root_features).rglob("*.pt")])
    kmeans.fit(features)
    inertia = -kmeans.score(features) / len(features)
    logger.info("K-means inertia: %s", inertia)
    return kmeans


def finetune_hubert(  # noqa: PLR0914
    name: str,
    project: str,
    workdir: Path,
    checkpoint: Path,
    manifest: str,
    *,
    n_clusters: int,
    target_layer: int,
) -> None:
    """Finetune HuBERT on DiscoPhon data with the default configuration.

    Args:
        name: Name of the run
        project: Wandb project
        workdir: Path to workdir
        checkpoint: Path to pretrained checkpoint
        manifest: Path to the manifest
        n_clusters: Number of clusters
        target_layer: Target layer
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

        # HuBERT data setup
        temp_manifest = stack.enter_context(NamedTemporaryFile(suffix=".jsonl"))
        temp_features = stack.enter_context(TemporaryDirectory(prefix="features-", dir=rundir))
        patch_manifest_with_paths(manifest, temp_manifest.name)
        kmeans = fit_kmeans_from_checkpoint(manifest, checkpoint, temp_features, target_layer, n_clusters, SEED)
        joblib.dump(kmeans, rundir / "kmeans.joblib")
        new_manifest = rundir / "manifest-with-units.jsonl"
        patch_manifest_with_units(temp_manifest.name, new_manifest, temp_features, kmeans)
        loader = build_dataloader_with_labels(hubert_ft_data_config(str(new_manifest)), MaskingConfig())

        # HuBERT
        model = HuBERTPretrain(n_clusters).to(device).train()
        state_dict = torch.load(checkpoint)["model"]
        del state_dict["logit_generator.label_embeddings"]
        model.load_state_dict(state_dict, strict=False)

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
        meters = AverageMeters(["loss", "grad_norm", "batch_size", "feature_loss"], device=device)
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
            for waveforms, labels, attn_mask, mask in loader:
                if step >= cfg.max_steps:
                    break
                with torch.autocast("cuda", dtype):
                    loss, outputs = model(
                        waveforms.to(device),
                        labels.to(device),
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
                    feature_loss=outputs["feature_loss"],
                )
                pbar.update()
                if step % LOG_INTERVAL == 0:
                    infos = meters.pop() | {"lr": lr, "step": step, "epoch": epoch}
                    wandb.log(infos)
                    pbar.set_postfix(loss=infos["loss"], feature_loss=infos["feature_loss"])
                ckpt.save(step, epoch)
                profiler.step()
        ckpt.save_final(step, epoch)
