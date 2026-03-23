import argparse
import json
from pathlib import Path

import polars as pl
from datasets import disable_progress_bars, load_dataset
from torchcodec.encoders import AudioEncoder
from tqdm import tqdm

from discophon.data import SAMPLE_RATE, read_rttm
from discophon.prepare import split_across_slurm_array


def segment_dataset(path_dataset: str, path_rttm: str, path_output: str, *, num_zeros: int = 5) -> None:
    disable_progress_bars()
    idx_start, idx_end = split_across_slurm_array(len(load_dataset(path=path_dataset, split="train")))
    output = Path(path_output)
    print(f"Segment files between indices [{idx_start}, {idx_end}[.")
    dataset = load_dataset(path=path_dataset, split=f"train[{idx_start}:{idx_end}]")

    unvoiced = []
    rttm = read_rttm(path_rttm).with_columns((pl.col("Turn Onset") + pl.col("Turn Duration")).alias("Turn Offset"))
    for data in tqdm(dataset, desc="Segmentation"):
        segments = rttm.filter(pl.col("File ID") == data["id"])
        if segments.height == 0:
            unvoiced.append(data["id"])
        for i, (onset, offset) in enumerate(segments.sort("Turn Onset")[["Turn Onset", "Turn Offset"]].iter_rows()):
            dest = output / data["iso3"] / (data["id"] + f"_{i:0{num_zeros}d}.wav")
            dest.parent.mkdir(parents=True, exist_ok=True)
            samples = data["audio"].get_samples_played_in_range(onset, offset)
            if samples.sample_rate != SAMPLE_RATE:
                raise ValueError(data["id"])
            AudioEncoder(samples.data, sample_rate=samples.sample_rate).to_file(dest)
    if unvoiced:
        print("Unvoiced files:\n", json.dumps(unvoiced, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Segment dataset")
    parser.add_argument("path", help="Path to the HuggingFace dataset")
    parser.add_argument("rttm", help="Path to the RTTM file")
    parser.add_argument("output", help="Output path")
    parser.add_argument("--num-zeros", type=int, default=5)
    args = parser.parse_args()
    segment_dataset(args.path, args.rttm, args.output, num_zeros=args.num_zeros)
