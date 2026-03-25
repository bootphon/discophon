"""Download and prepare the DiscoPhon benchmark dataset."""

import math
import os
import tarfile
from pathlib import Path
from typing import Literal, get_args

import fsspec
import polars as pl
import soundfile as sf
import soxr
from tqdm import tqdm

from discophon.data import SAMPLE_RATE, Splits
from discophon.languages import ISO6393_TO_CV, commonvoice_languages, get_language

__all__ = ["download_benchmark", "prepare_commonvoice_datasets"]


def split_across_slurm_array(n_total: int) -> tuple[int, int]:
    if "SLURM_NTASKS" not in os.environ:
        return 0, n_total
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        array_id, num_arrays = int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        if os.environ["SLURM_ARRAY_TASK_MIN"] != "0" or int(os.environ["SLURM_ARRAY_TASK_MAX"]) != num_arrays - 1:
            raise ValueError(
                f"Inside a SLURM array, but {os.environ['SLURM_ARRAY_TASK_MIN']=} and "
                f"{os.environ['SLURM_ARRAY_TASK_MAX']=} are not consistent with "
                f"{os.environ["SLURM_ARRAY_TASK_COUNT"]=}."
            )
    else:
        array_id, num_arrays = 0, 1
    n_per_array = math.ceil(n_total / num_arrays)
    start = array_id * n_per_array
    end = min(start + n_per_array, n_total)
    return start, end


def download_file(url: str, dest: str | Path, *, chunk_size: int = 2**20) -> None:
    with (
        fsspec.open(url, "rb") as src,
        Path(dest).open("wb") as dst,
        tqdm(total=src.size, unit_scale=True, unit_divisor=1024, unit="B") as progress,
    ):
        while chunk := src.read(chunk_size):
            dst.write(chunk)
            progress.update(len(chunk))


def download_benchmark(path_dataset: str | Path) -> None:
    """Download and extract the DiscoPhon dataset.

    Arguments:
        path_dataset: Target path to the DiscoPhon dataset.
    """
    path_dataset = Path(path_dataset)
    path_dataset.mkdir(exist_ok=True, parents=True)
    download_file("https://cognitive-ml.fr/downloads/discophon/benchmark.tar.gz", path_dataset / "benchmark.tar.gz")
    with tarfile.open(path_dataset / "benchmark.tar.gz", "r:gz") as tar:
        tar.extractall(path_dataset, filter="data")
    (path_dataset / "benchmark.tar.gz").unlink()


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


def get_filenames(manifests: Path, iso_code: str, *, split: Splits) -> list[str]:
    if split not in get_args(Splits):
        raise ValueError(f"Invalid {split=}. Must be in {get_args(Splits)}")
    if split != "all":
        manifest = pl.read_csv(manifests / f"manifest-{iso_code}-{split}.csv")
    else:
        manifest = pl.concat([pl.read_csv(path) for path in manifests.glob(f"manifest-{iso_code}-*.csv")])
    return sorted(manifest["fileid"].unique().to_list())


def prepare_commonvoice_datasets(path_dataset: str | Path, language: str) -> None:
    """Prepare the Common Voice datasets needed for DiscoPhon by resampling and copying the audio files.

    The specific Common Voice data should exist in `path_dataset/raw`: the audio files are expected to be
    in `path_dataset/raw/${cv_code}/clips` where `cv_code` is the Common Voice specific language code of `language`.

    Arguments:
        path_dataset: Path to the DiscoPhon dataset.
        language: Name of the language of the Common Voice dataset under consideration.
                  Also works with ISO-639-3 code or Common Voice code.
    """
    iso_code = get_language(language).iso_639_3
    src = Path(path_dataset) / "raw" / ISO6393_TO_CV[iso_code] / "clips"
    dest = Path(path_dataset) / "audio" / iso_code / "all"
    if not src.is_dir():
        raise ValueError(f"Directory {src} does not exist.")
    dest.mkdir(exist_ok=True, parents=True)
    filenames = get_filenames(Path(path_dataset) / "manifest", iso_code, split="all")
    filenames = filenames[slice(split_across_slurm_array(len(filenames)))]
    for filename in tqdm(filenames, desc="Resampling and converting to WAV"):
        resample(
            src / Path(filename).with_suffix(".mp3"),
            dest / Path(filename).with_suffix(".wav"),
            output_sample_rate=SAMPLE_RATE,
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Prepare Phoneme Discovery benchmark")
    subparsers = parser.add_subparsers(dest="command", required=True, help="command to run")
    parser_download = subparsers.add_parser(
        "download",
        description="Download benchmark data",
        help="download benchmark data",
    )
    parser_download.add_argument("data", help="path to data directory", type=Path)
    parser_audio = subparsers.add_parser("audio", description="Prepare audio files", help="prepare audio files")
    parser_audio.add_argument("data", help="path to data directory", type=Path)
    parser_audio.add_argument(
        "code",
        help="CommonVoice language ISO 639-3 code",
        choices=[lang.iso_639_3 for lang in commonvoice_languages()],
    )
    args = parser.parse_args()
    match args.command:
        case "download":
            download_benchmark(args.data)
        case "audio":
            prepare_commonvoice_datasets(args.data, args.code)
        case _:
            parser.error("Invalid command")
