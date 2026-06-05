import math
from collections.abc import Iterable
from pathlib import Path

import polars as pl
import torch
from sklearn.cluster import MiniBatchKMeans
from spidr.config import SAMPLE_RATE, DataConfig, OptimizerConfig
from spidr.data import read_manifest
from torch.nn import functional as F
from torch.optim import Optimizer, lr_scheduler
from torch.utils.data import Dataset
from torchcodec.decoders import WavDecoder
from tqdm import tqdm

from discophon.data import manifest_filename
from discophon.languages import get_language

SEED = 0
SAVE_INTERVAL = 1_000
LOG_INTERVAL = 200


class DiscophonAudioDataset(Dataset):
    def __init__(self, root: Path | str, language: str, split: str, *, normalize: bool) -> None:
        super().__init__()
        self.normalize = normalize
        self.root, self.language, self.split = Path(root), get_language(language), split
        self.manifest = read_manifest(self.root / "manifest" / manifest_filename(self.language, self.split))

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        fileid = self.manifest[index, "fileid"]
        path = self.root / "audio" / self.language.iso_639_3 / self.split / f"{fileid}.wav"
        samples = WavDecoder(path).get_all_samples()
        waveform = samples.data
        if samples.sample_rate != SAMPLE_RATE or waveform.size(0) != 1:
            raise ValueError(index)
        if self.normalize:
            waveform = F.layer_norm(waveform, waveform.shape)
        return fileid, waveform.squeeze()


def ft_optimizer_config() -> OptimizerConfig:
    return OptimizerConfig(
        lr=5e-5,
        weight_decay=0.01,
        betas=(0.9, 0.95),
        max_steps=20_000,
        eps=1e-6,
        max_norm=10.0,
        warmup_steps=600,
        hold_steps=9_400,
        decay_steps=10_000,
    )


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


def spidr_ft_data_config(manifest: str) -> DataConfig:
    return DataConfig(
        manifest,
        min_sample_size=0,
        max_sample_size=320_000,
        max_batch_length=3_800_000,
        num_buckets=5,
        num_workers=10,
        prefetch_factor=2,
        bucket_method="percentile",
    )


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


def patch_manifest_with_paths(src: str | Path, dest: str | Path) -> None:
    manifest = read_manifest(src)
    if "path" not in manifest.columns:
        discophon = Path(src).parent.parent.resolve()
        _, lang, *split = Path(src).stem.split("-")
        audios = discophon / "audio" / lang / "-".join(split)
        if not audios.is_dir():
            raise ValueError(audios)
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


def get_target_layers(layers: int | Iterable[int] | None, available: Iterable[int]) -> set[int]:
    available = set(available)
    if layers is None:
        return available
    if isinstance(layers, int):
        if layers in available:
            return {layers}
        raise ValueError(f"Invalid layer: {layers}")
    layers = set(layers)
    if layers & available != layers:
        raise ValueError(f"Some of the requested layers ({layers}) are not available (choose among: {available})")
    return layers
