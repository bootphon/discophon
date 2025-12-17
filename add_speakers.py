from pathlib import Path
from pprint import pprint

import polars as pl

paths = {
    "cmn": Path("/store/data/raw_data/commonvoice/cv17/cv17_zh-CN"),
    "eus": Path("/store/data/raw_data/commonvoice/cv-corpus-22.0-2025-06-20/eu"),
    "jpn": Path("/store/data/raw_data/commonvoice/cv-corpus-22.0-2025-06-20/ja"),
    "swa": Path("/mnt/legacy_nas1/data/raw_data/commonvoice/cv23/audio/swahili"),
    "tam": Path("/mnt/legacy_nas1/data/raw_data/commonvoice/cv23/audio/tamil"),
    "tha": Path("/mnt/legacy_nas1/data/raw_data/commonvoice/cv23/audio/thai"),
    "tur": Path("/mnt/legacy_nas1/data/raw_data/commonvoice/cv23/audio/turkish"),
    "ukr": Path("/mnt/legacy_nas1/data/raw_data/commonvoice/cv23/audio/ukranian"),
}


def read_all(path: Path) -> pl.DataFrame:
    return pl.read_csv(path / "validated.tsv", separator="\t", quote_char=None)


def merge(ref: pl.DataFrame, df: pl.DataFrame) -> pl.DataFrame:
    return df.join(
        ref.select("client_id", "path")
        .with_columns(pl.col("path").str.strip_suffix(".mp3").str.split("/").list.get(-1))
        .rename({"client_id": "speaker", "path": "fileid"}),
        on="fileid",
        how="left",
    ).select("fileid", "speaker", "num_samples")


if __name__ == "__main__":
    for path in [Path("/store/projects/phoneme_discovery/benchmark/manifest/manifest-tha-test.csv")]:
        code = path.stem.split("-")[1]
        manifest = pl.read_csv(path)
        if "speaker" in manifest.columns:
            continue
        print(path.stem)
        reference = read_all(paths[code])
        new_manifest = merge(reference, manifest)
        if not (len(new_manifest.drop_nulls()) == len(new_manifest) == len(manifest)):
            pprint(new_manifest.filter(pl.col("speaker").is_null())["fileid"].to_list())
        else:
            new_manifest.write_csv(path)
