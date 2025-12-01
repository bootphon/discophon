import zipfile
from pathlib import Path
from typing import Literal, get_args

import httpx
import polars as pl
import soundfile as sf
import soxr
from discophon.core import COMMONVOICE_TO_ISO6393, SAMPLE_RATE, split_for_distributed
from tqdm import tqdm

Splits = Literal["all", "train-10min", "train-1h", "train-10h", "train-100h", "train-all", "dev", "test"]


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
    download("https://cognitive-ml.fr/downloads/phoneme-discovery/benchmark.zip", data / "benchmark.zip")
    with zipfile.ZipFile(data / "benchmark.zip") as myzip:
        myzip.extractall(data)
    (data / "benchmark.zip").unlink()
    download("https://cognitive-ml.fr/downloads/phoneme-discovery/wolof.zip", data / "wolof.zip")
    with zipfile.ZipFile(data / "wolof.zip") as myzip:
        myzip.extractall(data)
    (data / "wolof.zip").unlink()


def resample(
    inp: str | Path,
    output: str | Path,
    *,
    output_sample_rate: int,
    quality: Literal["vhq", "hq", "mq", "lq"] = "hq",
) -> None:
    audio, input_sample_rate = sf.read(inp)
    resampled = soxr.resample(audio, input_sample_rate, output_sample_rate, quality)
    sf.write(output, resampled, output_sample_rate)


def get_filenames(manifests: Path, iso_code: str, *, split: Splits) -> list[str]:
    if split not in get_args(Splits):
        raise ValueError(f"Invalid {split=}. Must be in {get_args(Splits)}")
    if split != "all":
        manifest = pl.read_csv(manifests / f"manifest-{iso_code}-{split}.csv")
    else:
        manifest = pl.concat([pl.read_csv(path) for path in manifests.glob(f"manifest-{iso_code}-*.csv")])
    return sorted(manifest["fileid"].unique().to_list())


def prepare_downloaded_benchmark(data: str | Path, code: str) -> None:
    if code not in COMMONVOICE_TO_ISO6393:
        raise ValueError(f"Invalid CommonVoice code: {code}. Available ones: {list(COMMONVOICE_TO_ISO6393)}")
    iso_code = COMMONVOICE_TO_ISO6393[code]
    src, dest = Path(data) / "raw" / code / "clips", Path(data) / "audio" / iso_code / "all"
    if not src.is_dir():
        raise ValueError(f"Directory {src} does not exist.")
    dest.mkdir(exist_ok=True, parents=True)
    filenames = get_filenames(Path(data) / "manifests", iso_code, split="all")
    filenames = split_for_distributed(filenames)
    for filename in tqdm(filenames, desc="Resampling and converting to WAV"):
        resample(
            src / Path(filename).with_suffix(".mp3"),
            dest / Path(filename).with_suffix(".wav"),
            output_sample_rate=SAMPLE_RATE,
        )
