# %%
from pathlib import Path

import polars as pl
import soundfile as sf

num_frames = {
    split: dict(
        sorted(
            {
                p.stem: sf.info(p).frames
                for p in Path(f"/store/projects/phoneme_discovery/benchmark/audio/wol/{split}").glob("*.wav")
            }.items()
        )
    )
    for split in ["dev", "test", "train-10min", "train-1h", "train-10h", "train-all"]
}

# %%
sum(num_frames["test"].values()) / 16_000 / 3600
# %%
for split, manifest in num_frames.items():
    df = pl.DataFrame(
        [
            {"fileid": k, "num_samples": v, "speaker": "WOL_" + k.removeprefix("WOL_").split("_")[0]}
            for k, v in manifest.items()
        ]
    ).select("fileid", "speaker", "num_samples")
    df.write_csv(
        f"/store/projects/phoneme_discovery/benchmark/manifests/manifest-wol-{split}.csv",
    )

# %%
speakers_in_splits = {
    n: {"WOL_" + name.removeprefix("WOL_").split("_")[0] for name in manifest} for n, manifest in num_frames.items()
}
names = {name for names in speakers_in_splits.values() for name in names}
# %%
len(names)
# %%
speakers = []
for name in sorted(names):
    splits = sorted([split for split, manifest in speakers_in_splits.items() if name in manifest])
    if any(s.startswith("train") for s in splits):
        assert "dev" not in splits and "test" not in splits
        assert "train-all" in splits
    speakers.append(
        {
            "language": "wol",
            "speaker": name,
            "gender": None,
            "split": splits,
        }
    )
pl.DataFrame(speakers).write_ndjson("/store/projects/phoneme_discovery/benchmark/manifests/speakers.jsonl")

# %%
num_frames
# %%
names
# %%
speakers
# %%
pl.DataFrame(speakers).write_ndjson("/store/projects/phoneme_discovery/benchmark/manifests/speakers.jsonl")
# %%
