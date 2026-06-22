"""HuBERT finetuning."""

import logging
from collections.abc import Iterable
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Literal

import joblib
import orjson
import torch
import wandb
from minimal_hubert import HuBERT, HuBERTPretrain
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
    DiscophonAudioDataset,
    ft_optimizer_config,
    get_target_layers,
    hubert_ft_data_config,
    patch_manifest_with_paths,
    patch_manifest_with_units,
    tristage_scheduler,
)
from discophon.data import units_filename

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


@torch.inference_mode()
def extract_hubert_discrete_units(
    path_dataset: str | Path,
    path_units: str | Path,
    language: str,
    split: Literal["dev", "test", "train-10min", "train-1h", "train-10h"],
    pretrained_model_name_or_path: str | Path,
    kmeans_by_layer: dict[int, MiniBatchKMeans],
    *,
    layers: int | Iterable[int] | None = None,
) -> None:
    """Extract HuBERT discrete units for all utterances of a DiscoPhon split.

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
        pretrained_model_name_or_path: HuBERT checkpoint or HuggingFace model identifier.
        kmeans_by_layer: Mapping from 1-based layer index to the K-means model used to
            quantize that layer.
        layers: Layers to extract. If `None`, all encoder layers are used. Only layers present
            in both `layers` and `kmeans_by_layer` are written.
    """
    path_units = Path(path_units)
    dataset = DiscophonAudioDataset(path_dataset, language, split, normalize=True)
    model = HuBERT.from_pretrained(pretrained_model_name_or_path).eval().cuda()
    layers = get_target_layers(layers, [i + 1 for i in range(len(model.encoder.layers))])
    for fileid, waveform in tqdm(dataset, desc=f"{dataset.language.iso_639_3}-{dataset.split}"):
        all_features = model.get_intermediate_outputs(waveform.unsqueeze(0).cuda())
        for layer, features in enumerate(all_features):
            if layer + 1 not in layers or layer + 1 not in kmeans_by_layer:
                continue
            units = kmeans_by_layer[layer + 1].predict(features.squeeze().cpu().numpy()).tolist()
            entry = {"file": fileid, "units": units}
            jsonl = path_units / f"{layer + 1}" / units_filename(dataset.language, dataset.split)
            jsonl.parent.mkdir(exist_ok=True, parents=True)
            with jsonl.open("ab") as f:
                f.write(orjson.dumps(entry, option=orjson.OPT_APPEND_NEWLINE))


@torch.inference_mode()
def extract_hubert_continuous_features(
    path_dataset: str | Path,
    path_features: str | Path,
    language: str,
    split: Literal["dev", "test", "train-10min", "train-1h", "train-10h"],
    pretrained_model_name_or_path: str | Path,
    *,
    layers: int | Iterable[int] | None = None,
) -> None:
    """Extract HuBERT continuous features for all utterances of a DiscoPhon split.

    For each requested layer, the features are saved as PyTorch tensors at
    `path_features / {layer} / {iso_639_3} / {split} / {fileid}.pt`.

    Args:
        path_dataset: Path to the DiscoPhon dataset.
        path_features: Output directory under which per-layer feature tensors are written.
        language: Language identifier resolved by [`get_language`][discophon.languages.get_language],
            either name or ISO 639-3 code.
        split: Dataset split to process.
        pretrained_model_name_or_path: HuBERT checkpoint or HuggingFace model identifier.
        layers: Layers to extract. If `None`, all encoder layers are used.
    """
    path_features = Path(path_features)
    dataset = DiscophonAudioDataset(path_dataset, language, split, normalize=True)
    model = HuBERT.from_pretrained(pretrained_model_name_or_path).eval().cuda()
    layers = get_target_layers(layers, [i + 1 for i in range(len(model.encoder.layers))])
    for fileid, waveform in tqdm(dataset, desc=f"{dataset.language.iso_639_3}-{dataset.split}"):
        all_features = model.get_intermediate_outputs(waveform.unsqueeze(0).cuda())
        for layer, features in enumerate(all_features):
            if layer + 1 not in layers:
                continue
            path = path_features / f"{layer + 1}" / dataset.language.iso_639_3 / dataset.split / f"{fileid}.pt"
            path.parent.mkdir(exist_ok=True, parents=True)
            torch.save(features.squeeze().cpu(), path)
