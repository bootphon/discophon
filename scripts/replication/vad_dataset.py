import argparse
import os
from pathlib import Path

import torch
from datasets import disable_progress_bars, load_dataset
from filelock import FileLock
from pyannote.audio import Model
from pyannote.audio.pipelines import VoiceActivityDetection
from tqdm import tqdm

from discophon.data import SAMPLE_RATE
from discophon.prepare import split_across_slurm_array


def vad_dataset(path_dataset: str, path_rttm: str, *, model: str, token: str | None = None) -> None:
    disable_progress_bars()
    lock, rttm = FileLock(f"{path_rttm}.lock"), Path(path_rttm)

    device = torch.device(f"cuda:{int(os.getenv('SLURM_LOCALID', '0'))}")
    segmentation = Model.from_pretrained(model, token=token)
    pipeline = VoiceActivityDetection(segmentation=segmentation).to(device)  # ty: ignore[invalid-argument-type]
    pipeline.instantiate({"min_duration_on": 0.0, "min_duration_off": 0.0})

    idx_start, idx_end = split_across_slurm_array(len(load_dataset(path=path_dataset, split="train")))
    print(f"VAD files between indices [{idx_start}, {idx_end}[.")
    dataset = load_dataset(path=path_dataset, split=f"train[{idx_start}:{idx_end}]")

    for data in tqdm(dataset, desc="VAD"):
        samples = data["audio"].get_all_samples()
        if samples.sample_rate != SAMPLE_RATE:
            raise ValueError(data["id"])
        vad = pipeline({"waveform": samples.data.to(device), "sample_rate": samples.sample_rate})
        vad.uri = data["id"]
        with lock, rttm.open("a", encoding="utf-8") as f:
            vad.write_rttm(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("path", type=Path, help="Path to the HuggingFace dataset")
    parser.add_argument("rttm", type=Path, help="Path to the output RTTM file")
    parser.add_argument("--model", type=str, help="Pyannote model", default="pyannote/segmentation-3.0")
    parser.add_argument("--token", type=str, help="HuggingFace token", default=os.getenv("HF_TOKEN"))
    args = parser.parse_args()
    vad_dataset(args.path, args.rttm, model=args.model, token=args.token)
