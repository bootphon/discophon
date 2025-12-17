# %%
import shutil
import subprocess
from pathlib import Path

from discophon.core import Language

root_src = Path("/mnt/legacy_nas1/projects/phoneme_discovery_benchmark/splits_improved")
dest = Path("/mnt/legacy_nas1/projects/phoneme_discovery/benchmark")


def copytree(src: Path, dest: Path) -> None:
    subprocess.run(["rsync", "-a", "-i", "-P", str(src) + "/", str(dest) + "/"], check=True)


# for code in ("sw", "ta", "th", "tr", "uk", "zh-CN"):
for code in ("th",):
    print(f"Processing language code: {code}")
    lang = Language.from_commonvoice(code)
    src = root_src / "dev-lang" / code if lang.split == "dev" else root_src / "test-lang" / code

    print("Copy alignments...")
    # Copy alignments
    for split in ["dev", "test"]:
        align = src / "alignment" / f"{code}-{split}.align"
        shutil.copy2(align, dest / "alignment" / f"alignment-{lang!s}-{split}.txt")

    print("Copy items...")
    # Copy item
    for split in ["dev", "test"]:
        for mine, manel in [("phoneme", "phonebase"), ("triphone", "triphone")]:
            item = src / "item" / f"{code}-{manel}-{split}.item"
            # item = src / "item" / f"{manel}-{code}-{split}.item"
            shutil.copy2(item, dest / "item" / f"{mine}-{lang!s}-{split}.item")

    # Copy dev and test audio:
    for split in ["dev", "test"]:
        print(f"Copy {split} audio...")
        audio = src / "audio" / split
        copytree(audio, dest / "audio" / str(lang) / "all")
        for wav in audio.glob("*.wav"):
            file = dest / "audio" / str(lang) / split / wav.name
            if not file.is_symlink():
                file.symlink_to("../all/" + wav.name)

    print("Copy train audio...")
    audio = src / "audio" / "train" / "train-10h"
    copytree(audio, dest / "audio" / str(lang) / "all")
    for split in ["train-10h", "train-1h", "train-10min"]:
        src_dir = src / "audio" / "train" / split
        if not src_dir.is_dir():
            continue
        print(f"Create symlinks for {split}...")
        split_dir = dest / "audio" / str(lang) / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for wav in src_dir.glob("*.wav"):
            file = split_dir / wav.name
            if not file.is_symlink():
                file.symlink_to("../all/" + wav.name)
# %%
