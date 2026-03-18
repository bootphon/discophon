"""Training loop."""

import logging
import math
from contextlib import ExitStack
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

import joblib
import polars as pl
import torch
import wandb
from minimal_hubert import HuBERTPretrain
from minimal_hubert.data import build_dataloader_with_labels
from minimal_hubert.features import compute_and_save_hubert_features
from minimal_hubert.kmeans import build_kmeans
from sklearn.cluster import MiniBatchKMeans
from spidr.checkpoint import Checkpointer
from spidr.config import DataConfig, MaskingConfig
from spidr.data import read_manifest
from spidr.environment import set_seed, setup_environment, setup_pytorch
from spidr.tools import AverageMeters, profiler_context
from torch import GradScaler
from torch.nn.utils import clip_grad_norm_
from torch.optim import AdamW, Optimizer, lr_scheduler
from tqdm import tqdm

logger = logging.getLogger()


def tristage_scheduler(
    opt: Optimizer,
    *,
    warmup_steps: int,
    hold_steps: int,
    decay_steps: int,
    init_lr_scale: float = 1e-2,
    final_lr_scale: float = 1e-2,
) -> lr_scheduler.SequentialLR:
    warmup = lr_scheduler.LinearLR(opt, start_factor=init_lr_scale, total_iters=warmup_steps)
    hold = lr_scheduler.LinearLR(opt, start_factor=1.0, total_iters=hold_steps)
    decay = lr_scheduler.LambdaLR(opt, lambda step: math.exp(math.log(final_lr_scale) * step / decay_steps))
    return lr_scheduler.SequentialLR(opt, [warmup, hold, decay], [warmup_steps, hold_steps + warmup_steps])


def hubert_ft_data_config(manifest: str) -> DataConfig:
    return DataConfig(
        manifest,
        min_sample_size=0,
        max_sample_size=250_000,
        max_batch_length=2_800_000,
        num_buckets=5,
        num_workers=10,
        prefetch_factor=2,
        bucket_method="percentile",
    )


def patch_manifest_with_paths(src: str | Path, dest: str | Path) -> None:
    manifest = read_manifest(src)
    if "path" not in manifest.columns:
        discophon = Path(src).parent.parent.resolve()
        _, lang, *split = Path(src).stem.split("-")
        audios = discophon / "audio" / lang / "-".join(split)
        assert audios.is_dir()
        manifest = manifest.with_columns(path=pl.concat_str(pl.lit(str(audios) + "/"), "fileid", pl.lit(".wav")))
    manifest.write_csv(dest)


def patch_manifest_with_units(
    src: str | Path,
    dest: str | Path,
    root_features: str | Path,
    kmeans: MiniBatchKMeans,
) -> None:
    root = Path(root_features)
    manifest = read_manifest(src)
    new_manifest = []
    for row in tqdm(manifest.iter_rows(named=True), total=manifest.height):
        features = torch.load(root / f"{row['fileid']}.pt")
        units = kmeans.predict(features.numpy()).tolist()
        new_manifest.append(row | {"units": units})
    pl.DataFrame(new_manifest).write_ndjson(dest)


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
    max_steps, seed = 20_000, 0
    with ExitStack() as stack:
        set_seed(seed)
        setup_pytorch(use_deterministic=False)
        setup_environment()
        rundir = workdir / project / name
        rundir.mkdir(parents=True, exist_ok=True)
        temp_manifest = stack.enter_context(NamedTemporaryFile(suffix=".jsonl"))
        temp_features = stack.enter_context(TemporaryDirectory(prefix="features-", dir=rundir))
        patch_manifest_with_paths(manifest, temp_manifest.name)
        kmeans = fit_kmeans_from_checkpoint(manifest, checkpoint, temp_features, target_layer, n_clusters, seed)
        joblib.dump(kmeans, rundir / "kmeans.joblib")
        new_manifest = rundir / "manifest-with-units.jsonl"
        patch_manifest_with_units(temp_manifest.name, new_manifest, temp_features, kmeans)

        wandb.init(project=project, name=name, mode="offline", dir=workdir)
        stack.callback(wandb.finish)
        device = torch.device("cuda")
        model = HuBERTPretrain(256).to(device).train()
        state_dict = torch.load(checkpoint)["model"]
        del state_dict["logit_generator.label_embeddings"]
        model.load_state_dict(state_dict, strict=False)
        optimizer = AdamW(model.parameters(), lr=5e-5, weight_decay=0.01, betas=(0.9, 0.95), eps=1e-6, fused=True)
        scaler = GradScaler("cuda")
        scheduler = tristage_scheduler(optimizer, warmup_steps=600, hold_steps=9_400, decay_steps=10_000)
        loader = build_dataloader_with_labels(hubert_ft_data_config(str(new_manifest)), MaskingConfig())
        ckpt = Checkpointer(rundir, 1_000)
        ckpt.init_state(model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler)
        step, epoch = int(ckpt.step), int(ckpt.epoch)
        stack.callback(lambda: ckpt.save(step, epoch))
        if torch.cuda.get_device_capability() >= (8, 0):
            model.compile(dynamic=True)
            dtype = torch.bfloat16
        else:
            dtype = torch.float16
        meters = AverageMeters(["loss", "grad_norm", "batch_size", "feature_loss"], device=device)
        profiler = stack.enter_context(profiler_context(rundir / "trace.html"))
        pbar = stack.enter_context(tqdm(total=max_steps, initial=step))
        while step < max_steps:
            epoch += 1
            loader.batch_sampler.set_epoch(epoch)  # ty: ignore[unresolved-attribute]
            for waveforms, labels, attn_mask, mask in loader:
                if step >= max_steps:
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
                grad_norm = clip_grad_norm_(model.parameters(), 10.0)
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
                if step % 200 == 0:
                    infos = meters.pop() | {"lr": lr, "step": step, "epoch": epoch}
                    wandb.log(infos)
                    pbar.set_postfix(loss=infos["loss"], feature_loss=infos["feature_loss"])
                ckpt.save(step, epoch)
                profiler.step()
        ckpt.save_final(step, epoch)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("name", type=str)
    parser.add_argument("project", type=str)
    parser.add_argument("workdir", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("manifest", type=str)
    parser.add_argument("--n-clusters", type=int, required=True)
    parser.add_argument("--layer", type=int, required=True)
    args = parser.parse_args()

    finetune_hubert(
        args.name,
        args.project,
        args.workdir,
        args.checkpoint,
        args.manifest,
        n_clusters=args.n_clusters,
        target_layer=args.layer,
    )
