import csv
from pathlib import Path

import soundfile as sf
from tqdm import tqdm

from .utils import MyPathLike


def write_manifest(
        dataset: MyPathLike,
        output: MyPathLike,
        file_extension: str = ".wav"
) -> None:
    lines = [Path(dataset).resolve().as_posix()]
    paths = list(Path(dataset).rglob(f"*{file_extension}"))
    for name in tqdm(paths):
        lines.append(f"{name.relative_to(dataset)}\t{sf.info(name).frames}")
    with open(output, "w") as f:
        f.write("\n".join(lines) + "\n")


def read_manifest(file_path: MyPathLike) -> dict[str, tuple[Path, int]]:
    manifest = {}
    with open(file_path, "r", newline="") as fp:
        reader = csv.reader(fp, delimiter="\t")
        root = Path(next(reader)[0])
        for row in reader:
            assert len(row) == 2, f"Invalid tsv file: {file_path}"
            file, num_samples = root / row[0], int(row[1])
            assert file.stem not in manifest, f"Duplicate file id: {file.stem}"
            manifest[file.stem] = (file, num_samples)
    return manifest


def read_alignment(path: MyPathLike, sep: str = " ") -> dict[str, list[str]]:
    phones = {}
    with open(path, "r", newline="") as fp:
        reader = csv.reader(fp, delimiter="\t")
        for row in reader:
            assert len(row) == 2
            phones[row[0]] = row[1].split(sep)
    return phones
