import math
import os
import tarfile
from collections.abc import Sequence
from pathlib import Path
from typing import Literal, get_args

import httpx
import polars as pl
import soundfile as sf
import soxr
from tqdm import tqdm

from discophon.core import SAMPLE_RATE
from discophon.core.languages import ISO6393_TO_CV


def split_for_distributed[T](sequence: Sequence[T]) -> Sequence[T]:
    if "SLURM_NTASKS" not in os.environ:
        return sequence
    array_id, num_arrays = int(os.getenv("SLURM_ARRAY_TASK_ID", "0")), int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        assert os.environ["SLURM_ARRAY_TASK_MIN"] == "0"
        assert int(os.environ["SLURM_ARRAY_TASK_MAX"]) == num_arrays - 1
    n_total = len(sequence)
    files_per_array = math.ceil(n_total / num_arrays)
    start = array_id * files_per_array
    end = min(start + files_per_array, n_total)
    return sequence[start:end]


def download(url: str, dest: str | Path) -> None:
    with Path(dest).open("wb") as download_file, httpx.stream("GET", url) as response:
        total, name = int(response.headers["Content-Length"]), Path(response.url.path).name
        with tqdm(desc=f"Downloading {name}", total=total, unit_scale=True, unit_divisor=1024, unit="B") as progress:
            num_bytes_downloaded = response.num_bytes_downloaded
            for chunk in response.iter_bytes():
                download_file.write(chunk)
                progress.update(response.num_bytes_downloaded - num_bytes_downloaded)
                num_bytes_downloaded = response.num_bytes_downloaded


def download_benchmark(data: str | Path) -> None:
    data = Path(data)
    data.mkdir(exist_ok=True, parents=True)
    url = "https://cognitive-ml.fr/downloads/phoneme-discovery/"
    download(url + "benchmark-assets.tar.gz", data / "benchmark-assets.tar.gz")
    with tarfile.open(data / "benchmark-assets.tar.gz", "r:gz") as tar:
        tar.extractall(data)
    (data / "benchmark-assets.tar.gz").unlink()
    download(url + "benchmark-audio.tar.gz", data / "benchmark-audio.tar.gz")
    with tarfile.open(data / "benchmark-audio.tar.gz", "r:gz") as tar:
        tar.extractall(data)
    (data / "benchmark-audio.tar.gz").unlink()


def resample(
    inp: str | Path,
    output: str | Path,
    *,
    output_sample_rate: int,
    quality: Literal["vhq", "hq", "mq", "lq"] = "vhq",
) -> None:
    audio, input_sample_rate = sf.read(inp)
    resampled = soxr.resample(audio, input_sample_rate, output_sample_rate, quality)
    sf.write(output, resampled, output_sample_rate)


Splits = Literal["all", "train-10min", "train-1h", "train-10h", "train-100h", "train-all", "dev", "test"]


def get_filenames(manifests: Path, iso_code: str, *, split: Splits) -> list[str]:
    if split not in get_args(Splits):
        raise ValueError(f"Invalid {split=}. Must be in {get_args(Splits)}")
    if split != "all":
        manifest = pl.read_csv(manifests / f"manifest-{iso_code}-{split}.csv")
    else:
        manifest = pl.concat([pl.read_csv(path) for path in manifests.glob(f"manifest-{iso_code}-*.csv")])
    return sorted(manifest["fileid"].unique().to_list())


def prepare_downloaded_benchmark(data: str | Path, iso_code: str) -> None:
    src, dest = (Path(data) / "raw" / ISO6393_TO_CV[iso_code] / "clips", Path(data) / "audio" / iso_code / "all")
    if not src.is_dir():
        raise ValueError(f"Directory {src} does not exist.")
    dest.mkdir(exist_ok=True, parents=True)
    filenames = get_filenames(Path(data) / "manifest", iso_code, split="all")
    filenames = split_for_distributed(filenames)
    for filename in tqdm(filenames, desc="Resampling and converting to WAV"):
        resample(
            src / Path(filename).with_suffix(".mp3"),
            dest / Path(filename).with_suffix(".wav"),
            output_sample_rate=SAMPLE_RATE,
        )
