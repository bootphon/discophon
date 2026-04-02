import argparse
from pathlib import Path

import polars as pl

from discophon.data import SAMPLE_RATE, read_gold_annotations_as_dataframe
from discophon.languages import ISO6393_TO_CV


def fix_last_offset(path_manifest: Path, path_alignment: Path) -> pl.DataFrame:
    manifest = pl.read_csv(path_manifest)
    current = read_gold_annotations_as_dataframe(path_alignment)
    last = (
        current.filter(pl.col("offset") == pl.col("offset").max().over("#file"))
        .join(manifest, left_on="#file", right_on="fileid")
        .with_columns(duration=(pl.col("num_samples") / SAMPLE_RATE).cast(pl.Decimal(precision=8, scale=6)))
        .select("#file", "duration")
    )
    return (
        current.join(last, on="#file")
        .with_columns(
            offset=pl.when(pl.col("offset") == pl.col("offset").max().over("#file"))
            .then("duration")
            .otherwise("offset")
        )
        .with_columns((pl.col("offset") * 100).floor() / 100)
        .select("#file", "onset", "offset", "#phone")
    )


def fix_all(path_alignments: Path, path_manifests: Path, path_new_alignments: Path) -> None:
    path_new_alignments.mkdir(exist_ok=True)
    for lang in ISO6393_TO_CV:
        for split in ["dev", "test"]:
            fix_last_offset(
                path_manifests / f"manifest-{lang}-{split}.csv",
                path_alignments / f"alignment-{lang}-{split}.txt",
            ).write_csv(path_new_alignments / f"alignment-{lang}-{split}.txt", separator=" ")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("path_alignments", type=Path)
    parser.add_argument("path_manifests", type=Path)
    parser.add_argument("path_new_alignments", type=Path)
    args = parser.parse_args()
    fix_all(args.path_alignments, args.path_manifests, args.path_new_alignments)
