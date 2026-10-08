"""Download and prepare the DiscoPhon benchmark dataset."""

import argparse
import contextlib
import hashlib
import io
import os
import re
import sys
import tarfile
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import IO

import polars as pl
import requests
import soundfile as sf
import soxr
from datacollective import DatasetDetails, download_dataset, list_datasets
from tqdm import tqdm

from discophon.data import SAMPLE_RATE, manifest_filename
from discophon.languages import ISO6393_TO_CV, Language, commonvoice_languages, get_language

__all__ = ["MissingClipsError", "check_commonvoice", "download_benchmark", "prepare_commonvoice"]

BENCHMARK_URL = "https://cognitive-ml.fr/downloads/phoneme-discovery/discophon_data.tar.gz"
BENCHMARK_SHA256 = "358dbc61e9e74e9b922e2ae38ec989d3ad4e101c9fd57b60e2ea3c0332b2502c"
BENCHMARK_SIZE = 5374865887
MDC_API_URL = "https://mozilladatacollective.com/api"
COMMONVOICE_RELEASE = re.compile(r"^Common Voice Scripted Speech (\d+)\.(\d+) - ")


class MissingClipsError(ValueError):
    """Raised when a Common Voice release does not contain all the clips needed by DiscoPhon."""

    def __init__(self, missing: set[str], n_needed: int, release: str) -> None:
        self.missing = sorted(missing)
        super().__init__(
            f"\n{'!' * 80}\n"
            f"{len(missing)} of the {n_needed} clips needed by DiscoPhon are missing from {release}.\n"
            "The DiscoPhon dataset cannot be rebuilt from this release. Missing clips:\n"
            + "\n".join(f"  {clip}" for clip in self.missing)
            + f"\n{'!' * 80}"
        )


def download_benchmark(path_dataset: str | Path) -> None:
    """Download and extract the DiscoPhon assets: manifests, alignments, items, and the audio we distribute.

    Arguments:
        path_dataset: Target path to the DiscoPhon dataset.

    """
    path_dataset = Path(path_dataset)
    path_dataset.mkdir(exist_ok=True, parents=True)
    archive = path_dataset / "discophon_data.tar.gz"
    offset = archive.stat().st_size if archive.is_file() else 0  # Resume an interrupted download
    if offset < BENCHMARK_SIZE:
        with requests.get(BENCHMARK_URL, headers={"Range": f"bytes={offset}-"}, stream=True, timeout=60) as response:
            response.raise_for_status()
            if response.status_code != 206:  # Range not supported by the server, restart from scratch
                offset = 0
            with (
                archive.open("ab" if offset else "wb") as f,
                tqdm(
                    total=BENCHMARK_SIZE,
                    initial=offset,
                    unit_scale=True,
                    unit_divisor=1024,
                    unit="B",
                    desc="Downloading",
                ) as progress,
            ):
                for chunk in response.iter_content(2**20):
                    f.write(chunk)
                    progress.update(len(chunk))
    with archive.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    if checksum != BENCHMARK_SHA256:
        archive.unlink()
        raise ValueError(f"Checksum mismatch for {archive}: expected {BENCHMARK_SHA256}, got {checksum}.")
    with tarfile.open(archive, "r:gz") as tar:
        for member in tqdm(tar, desc="Extracting", unit=" files"):
            root, parts = member.name.split("/", 1)
            if root != "discophon_data":
                raise ValueError(f"Unexpected tarfile: root is {root} but should be 'discophon_data'")
            member.name = parts
            tar.extract(member, path=path_dataset, filter="data")
    archive.unlink()


def needed_clips(path_dataset: Path, language: Language) -> set[str]:
    """Return the identifiers of all the files of `language` listed in the manifests."""
    manifests = list((path_dataset / "manifest").glob(manifest_filename(language, "*")))
    if not manifests:
        raise FileNotFoundError(f"No manifest for {language.name} in {path_dataset / 'manifest'}. Download first.")
    return set(pl.concat([pl.read_csv(path) for path in manifests])["fileid"].to_list())


def latest_release(language: Language) -> DatasetDetails:
    """Find the latest Common Voice Scripted Speech release of `language` on Mozilla Data Collective."""
    cv_code = ISO6393_TO_CV[language.iso_639_3]
    releases = {}
    for dataset in list_datasets("Common Voice Scripted Speech", locale=cv_code, results_per_page=50).items:
        if dataset.locale == cv_code and (match := COMMONVOICE_RELEASE.match(dataset.name or "")):
            releases[int(match[1]), int(match[2])] = dataset
    if not releases:
        raise ValueError(f"No Common Voice Scripted Speech release found for locale {cv_code}.")
    return releases[max(releases)]


def download_session(release: DatasetDetails) -> dict:
    """Open a download session for `release` on Mozilla Data Collective, without downloading anything."""
    if not (api_key := os.environ.get("MDC_API_KEY")):
        raise ValueError("Missing API key. Set `MDC_API_KEY` to your Mozilla Data Collective key.")
    response = requests.post(
        f"{MDC_API_URL}/datasets/{release.id}/download",
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=60,
    )
    if response.status_code == 403:
        raise PermissionError(f"Access denied to {release.name} ({release.id}): {response.text}")
    response.raise_for_status()
    return response.json()


def inaccessible_releases(languages: list[str]) -> list[DatasetDetails]:
    """Return the latest Common Voice releases of `languages` whose terms have not been accepted."""
    denied = []
    for language in languages:
        release = latest_release(get_language(language))
        try:
            download_session(release)
        except PermissionError:
            denied.append(release)
    return denied


def mp3_to_wav(mp3: bytes, wav: Path) -> None:
    """Resample an MP3 clip to 16 kHz and write it to `wav`, atomically to resume safely if interrupted."""
    audio, sample_rate = sf.read(io.BytesIO(mp3))
    tmp = wav.with_suffix(".wav.part")
    sf.write(tmp, soxr.resample(audio, sample_rate, SAMPLE_RATE, "VHQ"), SAMPLE_RATE, format="WAV")
    tmp.replace(wav)


def stream_clips(archive: IO[bytes], wanted: set[str]) -> Iterator[tuple[str, bytes]]:
    """Read a Common Voice `.tar.gz` stream, and find the clips in `wanted`.

    Yields:
        The identifier and the MP3 content of each clip found.

    """
    with tarfile.open(fileobj=archive, mode="r|gz") as tar:
        for member in tqdm(tar, desc="Reading archive", unit=" files"):
            path = PurePosixPath(member.name)
            if not (member.isfile() and path.parent.name == "clips" and path.suffix == ".mp3" and path.stem in wanted):
                continue
            if (clip := tar.extractfile(member)) is not None:
                yield path.stem, clip.read()


def prepare_commonvoice(path_dataset: str | Path, language: str) -> None:
    """Download the latest Common Voice release of `language`, and convert the clips needed by DiscoPhon to WAV.

    The clips are resampled to 16 kHz directly from the archive, and written to `path_dataset/audio/${code}/all`.
    Requires the `MDC_API_KEY` environment variable, and to have accepted the release terms on Mozilla Data Collective.

    This function can be resumed if interrupted: the download restarts where it stopped, and existing WAV files
    are skipped. The archive is kept in `path_dataset/raw` until all clips are processed.

    Arguments:
        path_dataset: Path to the DiscoPhon dataset. Must already contain the manifests.
        language: Name, ISO 639-3 code, or Common Voice code of the language.

    Raises:
        MissingClipsError: If some clips listed in the manifests are not in the release. The other clips are converted.

    """
    path_dataset, resolved = Path(path_dataset), get_language(language)
    dest = path_dataset / "audio" / resolved.iso_639_3 / "all"
    needed = needed_clips(path_dataset, resolved)
    todo = {fileid for fileid in needed if not (dest / f"{fileid}.wav").is_file()}
    if not todo:
        return
    release = latest_release(resolved)
    archive = download_dataset(release.id, download_directory=str(path_dataset / "raw"))
    with archive.open("rb") as f:
        checksum = hashlib.file_digest(f, "sha256").hexdigest()
    if release.checksum is not None and checksum != release.checksum:
        archive.unlink()
        raise ValueError(f"Checksum mismatch for {archive}: expected {release.checksum}, got {checksum}.")
    dest.mkdir(exist_ok=True, parents=True)
    with archive.open("rb") as f:
        for fileid, mp3 in stream_clips(f, todo):
            mp3_to_wav(mp3, dest / f"{fileid}.wav")
            todo.remove(fileid)
            if not todo:
                break
    archive.unlink()
    with contextlib.suppress(OSError):  # Other languages may still be downloading there
        archive.parent.rmdir()
    if todo:
        raise MissingClipsError(todo, len(needed), release.name or release.id)


def check_commonvoice(path_dataset: str | Path, language: str) -> DatasetDetails:
    """Check that the latest Common Voice release of `language` contains all the clips needed by DiscoPhon.

    The release is streamed, and nothing is written to disk.
    Requires the `MDC_API_KEY` environment variable, and to have accepted the release terms on Mozilla Data Collective.

    Arguments:
        path_dataset: Path to the DiscoPhon dataset. Must already contain the manifests.
        language: Name, ISO 639-3 code, or Common Voice code of the language.

    Returns:
        The details of the checked release.

    Raises:
        MissingClipsError: If some clips listed in the manifests are not in the release.

    """
    resolved = get_language(language)
    needed = needed_clips(Path(path_dataset), resolved)
    release = latest_release(resolved)
    found = set()
    with requests.get(download_session(release)["downloadUrl"], stream=True, timeout=60) as response:
        response.raise_for_status()
        for fileid, _ in stream_clips(response.raw, needed):
            found.add(fileid)
            if found == needed:
                break
    if missing := needed - found:
        raise MissingClipsError(missing, len(needed), release.name or release.id)
    return release


def cli(argv: list[str] | None = None) -> None:
    """Command-line entry point for dataset download and preparation."""
    parser = argparse.ArgumentParser(description="Prepare the DiscoPhon benchmark data")
    subparsers = parser.add_subparsers(dest="command", required=True, help="command to run")
    parser_download = subparsers.add_parser(
        "download",
        description="Download the manifests, alignments, items, and the audio distributed with DiscoPhon",
        help="download the benchmark assets",
    )
    parser_download.add_argument("data", type=Path, help="path to data directory")
    parser_cv = subparsers.add_parser(
        "commonvoice",
        description="Download the latest Common Voice releases and convert the clips needed by DiscoPhon to WAV. "
        "Can be resumed if interrupted. Requires the MDC_API_KEY environment variable.",
        help="download and prepare Common Voice audio",
    )
    parser_cv.add_argument("data", type=Path, help="path to data directory")
    codes = [lang.iso_639_3 for lang in commonvoice_languages()]
    parser_cv.add_argument(
        "languages",
        nargs="*",
        choices=codes,
        metavar="LANG",
        help=f"languages, among {codes} (all if none given)",
    )
    parser_cv.add_argument(
        "--check-only",
        action="store_true",
        help="only check that the releases contain all the needed clips, by streaming them without writing to disk",
    )
    args = parser.parse_args(argv)
    match args.command:
        case "download":
            download_benchmark(args.data)
        case "commonvoice":
            languages = args.languages or codes
            if denied := inaccessible_releases(languages):
                sys.exit(
                    "Accept the terms of these releases on Mozilla Data Collective, then rerun:\n"
                    + "\n".join(f"  {release.name}: {release.datasetUrl}" for release in denied)
                )
            failures = []
            for code in languages:
                try:
                    if args.check_only:
                        print(f"{code}: all clips found in {check_commonvoice(args.data, code).name}")
                    else:
                        prepare_commonvoice(args.data, code)
                        print(f"{code}: all clips prepared")
                except MissingClipsError as error:
                    failures.append(f"{code}: {error}")
            if failures:
                sys.exit(
                    f"\n{'#' * 80}\nCOMMON VOICE FAILED for {len(failures)} language(s)\n{'#' * 80}\n"
                    + "\n".join(failures)
                )
        case _:
            parser.error("Invalid command")


if __name__ == "__main__":
    cli()
