# %%
import polars as pl
from discophon.core import read_gold_annotations
from fastabx.dataset import read_labels
from rich.console import Console
from pathlib import Path
from discophon.core import read_gold_annotations
from datetime import timedelta
import soundfile as sf
from rich.table import Table

console = Console()
SAMPLE_RATE = 16_000


def fmt(td: timedelta) -> str:
    mm, ss = divmod(td.total_seconds(), 60)
    hh, mm = divmod(mm, 60)
    s = "%d:%02d:%02d" % (hh, mm, ss)
    if td.microseconds:
        s = s + ".%06d" % td.microseconds
    return s


def duration(path: str | Path) -> timedelta:
    samples = 0
    for p in Path(path).glob("*.wav"):
        info = sf.info(p)
        samples += info.frames
        assert info.samplerate == SAMPLE_RATE, f"Expected {SAMPLE_RATE}Hz, got {info.samplerate} in {p}"
    return timedelta(seconds=samples / SAMPLE_RATE)


def read_item(path: str | Path) -> pl.DataFrame:
    return read_labels(path, "#file", "onset", "offset")


def find_split(path: Path) -> str:
    name = path.stem
    if "100h" in name:
        return "100h"
    if "10h" in name:
        return "10h"
    if "1h" in name:
        return "1h"
    if "10m" in name:
        return "10m"
    if "all" in name:
        return "all"
    raise ValueError(f"Could not find split in {name}")


def test_metadata(path: Path) -> None:
    renaming = {
        "Filename": "file",
        "Speaker_ID": "speaker",
        "Gender_Category": "gender",
        "Estimated_Duration_Seconds": "seconds",
    }
    df = (
        pl.read_csv(path)
        .select(renaming.keys())
        .rename(renaming)
        .group_by("speaker", maintain_order=True)
        .agg(
            pl.col("file").len().alias("n_utterances"),
            pl.col("gender").unique(),
            pl.sum("seconds"),
        )
        .with_columns(duration=pl.duration(seconds=pl.col("seconds")))
    )
    assert (df["gender"].list.len() == 1).all()
    df = df.with_columns(pl.col("gender").list.first())


def test_duration(audio: Path) -> None:
    durations = {"dev": duration(audio / "dev"), "test": duration(audio / "test")}
    for path in (audio / "train").iterdir():
        durations[f"train-{find_split(path)}"] = duration(path)
    table = Table()
    table.add_column("split", style="cyan")
    table.add_column("duration", style="magenta")
    for split, dur in durations.items():
        table.add_row(split, fmt(dur))
    console.print(table)


def test_language(root: Path) -> None:
    console.print(f"[bold blue]Verifying:[/bold blue] {root.stem}")
    code = root.stem
    for align in (root / "alignment").glob("*.align"):
        read_gold_annotations(align)
    test_duration(root / "audio")


def test_pipeline(root: Path) -> None:
    for language in root.iterdir():
        test_language(language)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    test_pipeline(args.root)
