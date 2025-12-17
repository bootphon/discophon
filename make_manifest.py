from pathlib import Path

import polars as pl
import soundfile as sf

if __name__ == "__main__":
    split = "dev"
    root = Path("/mnt/legacy_nas1/projects/phoneme_discovery/benchmark")
    lang = root / "audio" / "tha"
    print(lang, split)
    files = [{"fileid": p.stem, "num_samples": sf.info(p).frames} for p in sorted((lang / split).glob("*.wav"))]
    pl.DataFrame(files).write_csv(root / "manifest" / f"manifest-{lang.stem}-{split}.csv")
