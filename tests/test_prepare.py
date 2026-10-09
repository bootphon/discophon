"""Tests for the Common Voice preparation, with a fake Mozilla Data Collective serving a tiny release."""

import functools
import hashlib
import io
import re
import shutil
import tarfile
import threading
from collections.abc import Iterator
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import polars as pl
import pytest
import soundfile as sf
from datacollective import DatasetDetails

from discophon import prepare
from discophon.data import SAMPLE_RATE, manifest_filename
from discophon.languages import commonvoice_languages, get_language
from discophon.validate import validate_dataset_structure

from .test_validate import build_valid_dataset

SWAHILI = get_language("swa")
RELEASE = "Common Voice Scripted Speech 27.0 - Swahili"
EXISTING = b"already there"


def make_mp3(seconds: float = 0.5, sample_rate: int = 48_000) -> bytes:
    buffer = io.BytesIO()
    t = np.arange(int(seconds * sample_rate)) / sample_rate
    sf.write(buffer, 0.5 * np.sin(2 * np.pi * 440 * t), sample_rate, format="MP3")
    return buffer.getvalue()


def make_release(path: Path, fileids: list[str]) -> Path:
    """Tiny Common Voice archive, with the same layout as the real ones, and some unrelated files."""
    with tarfile.open(path, "w:gz") as tar:
        for name, content in [
            ("cv-corpus/sw/validated.tsv", b"path\n"),
            *[(f"cv-corpus/sw/clips/{fileid}.mp3", make_mp3()) for fileid in fileids],
            ("cv-corpus/sw/clips/unrelated.mp3", make_mp3()),
        ]:
            info = tarfile.TarInfo(name)
            info.size = len(content)
            tar.addfile(info, io.BytesIO(content))
    return path


def write_manifests(root: Path, fileids: list[str]) -> None:
    (root / "manifest").mkdir(parents=True, exist_ok=True)
    pl.DataFrame({"fileid": fileids[:1]}).write_csv(root / "manifest" / manifest_filename(SWAHILI, "dev"))
    for split in ["test", "train-10h", "train-1h", "train-10min"]:
        pl.DataFrame({"fileid": fileids[1:]}).write_csv(root / "manifest" / manifest_filename(SWAHILI, split))


class FakeMDC:
    """Stand-in for the datacollective functions used by `discophon.prepare`."""

    def __init__(self, archive: Path, *, checksum: str | None = None) -> None:
        self.archive = archive
        self.checksum = checksum or hashlib.sha256(archive.read_bytes()).hexdigest()
        self.downloads = 0

    def list_datasets(self, *_: object, **__: object) -> SimpleNamespace:
        """Catalog with the release, an older one, and unrelated datasets."""
        return SimpleNamespace(
            items=[
                DatasetDetails(id="old", name="Common Voice Scripted Speech 9.0 - Swahili", locale="sw"),
                DatasetDetails(id="new", name=RELEASE, locale="sw", checksum=self.checksum),
                DatasetDetails(id="spontaneous", name="Common Voice Spontaneous Speech 99.0 - Swahili", locale="sw"),
                DatasetDetails(id="other", name="Common Voice Scripted Speech 99.0 - Hausa", locale="ha"),
            ]
        )

    def download_dataset(self, dataset_id: str, download_directory: str) -> Path:
        """Copy the archive, unless already there (like a resumed download)."""
        assert dataset_id == "new"
        self.downloads += 1
        target = Path(download_directory) / self.archive.name
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(self.archive, target)
        return target


@pytest.fixture(autouse=True)
def fake_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """Never use the API key of the environment."""
    monkeypatch.setenv("MDC_API_KEY", "key")


@pytest.fixture
def mdc(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeMDC:
    fake = FakeMDC(make_release(tmp_path / "release.tar.gz", ["a", "b", "c"]))
    monkeypatch.setattr(prepare, "list_datasets", fake.list_datasets)
    monkeypatch.setattr(prepare, "download_dataset", fake.download_dataset)
    return fake


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    root = build_valid_dataset(tmp_path / "dataset")
    write_manifests(root, ["a", "b", "c"])
    return root


@pytest.mark.usefixtures("mdc")
def test_latest_release_picks_highest_scripted_version() -> None:
    assert prepare.latest_release(SWAHILI).id == "new"
    with pytest.raises(ValueError, match="No Common Voice"):
        prepare.latest_release(get_language("jpn"))


def test_needed_clips_requires_manifests(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Download first"):
        prepare.needed_clips(tmp_path, SWAHILI)


def test_prepare_commonvoice_converts_needed_clips(dataset: Path, mdc: FakeMDC) -> None:
    prepare.prepare_commonvoice(dataset, "swa")
    wavs = dataset / "audio" / "swa" / "all"
    assert sorted(p.name for p in wavs.iterdir()) == ["a.wav", "b.wav", "c.wav"]
    info = sf.info(wavs / "a.wav")
    assert info.samplerate == SAMPLE_RATE
    assert info.frames == pytest.approx(0.5 * SAMPLE_RATE, abs=0.05 * SAMPLE_RATE)  # MP3 adds some padding
    assert not (dataset / "raw").exists()
    validate_dataset_structure(dataset)

    prepare.prepare_commonvoice(dataset, "swa")
    assert mdc.downloads == 1, "everything is already prepared, nothing should be downloaded"


@pytest.mark.usefixtures("mdc")
def test_prepare_commonvoice_skips_existing_wavs(dataset: Path) -> None:
    existing = dataset / "audio" / "swa" / "all" / "b.wav"
    existing.write_bytes(EXISTING)
    prepare.prepare_commonvoice(dataset, "swa")
    assert existing.read_bytes() == EXISTING
    assert (dataset / "audio" / "swa" / "all" / "c.wav").is_file()


@pytest.mark.usefixtures("mdc")
def test_prepare_commonvoice_resumes_after_interruption(dataset: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write = sf.write

    def write_once_then_interrupt(path: Path, *args: object, **kwargs: object) -> None:
        if any((dataset / "audio" / "swa" / "all").glob("*.wav")):
            raise KeyboardInterrupt
        write(path, *args, **kwargs)

    monkeypatch.setattr(prepare.sf, "write", write_once_then_interrupt)
    with pytest.raises(KeyboardInterrupt):
        prepare.prepare_commonvoice(dataset, "swa")
    assert len(list((dataset / "audio" / "swa" / "all").glob("*.wav"))) == 1
    assert (dataset / "raw" / "release.tar.gz").is_file(), "the archive must be kept to resume"

    monkeypatch.setattr(prepare.sf, "write", write)
    prepare.prepare_commonvoice(dataset, "swa")
    assert sorted(p.name for p in (dataset / "audio" / "swa" / "all").glob("*.wav")) == ["a.wav", "b.wav", "c.wav"]
    assert not (dataset / "raw").exists()


@pytest.mark.usefixtures("mdc")
def test_prepare_commonvoice_fails_on_missing_clips(dataset: Path) -> None:
    write_manifests(dataset, ["a", "b", "c", "gone", "lost"])
    with pytest.raises(prepare.MissingClipsError, match="2 of the 5 clips needed by DiscoPhon are missing") as error:
        prepare.prepare_commonvoice(dataset, "swa")
    assert error.value.missing == ["gone", "lost"]
    assert sorted(p.name for p in (dataset / "audio" / "swa" / "all").iterdir()) == ["a.wav", "b.wav", "c.wav"]


def test_prepare_commonvoice_rejects_corrupted_archive(
    dataset: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fake = FakeMDC(make_release(tmp_path / "release.tar.gz", ["a", "b", "c"]), checksum="0" * 64)
    monkeypatch.setattr(prepare, "list_datasets", fake.list_datasets)
    monkeypatch.setattr(prepare, "download_dataset", fake.download_dataset)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        prepare.prepare_commonvoice(dataset, "swa")
    assert not (dataset / "raw" / "release.tar.gz").exists()
    assert not any((dataset / "audio" / "swa" / "all").iterdir())


class QuietHandler(SimpleHTTPRequestHandler):
    """Static file server that does not log requests."""

    def log_message(self, format: str, *args: object) -> None:  # ruff: ignore[builtin-argument-shadowing]
        """Do not log requests."""


class RangeHandler(QuietHandler):
    """Static file server that honors `Range: bytes={start}-` requests, like the server hosting the benchmark."""

    honor_range: ClassVar[bool] = True
    requested_ranges: ClassVar[list[str]] = []

    def do_GET(self) -> None:
        """Send the requested range of the file if any, otherwise the whole file."""
        requested = self.headers.get("Range", "")
        self.requested_ranges.append(requested)
        if not (self.honor_range and (match := re.fullmatch(r"bytes=(\d+)-", requested))):
            super().do_GET()
            return
        data = Path(self.translate_path(self.path)).read_bytes()[int(match[1]) :]
        self.send_response(206)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture
def served_release(mdc: FakeMDC, monkeypatch: pytest.MonkeyPatch) -> Iterator[SimpleNamespace]:
    """Serve the release over HTTP, and answer the download session request like Mozilla Data Collective.

    Yields:
        The download session response, to modify its status code.

    """
    handler = functools.partial(QuietHandler, directory=mdc.archive.parent)
    server = ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    session = SimpleNamespace(
        status_code=200,
        text="",
        raise_for_status=lambda: None,
        json=lambda: {"downloadUrl": f"http://127.0.0.1:{server.server_port}/{mdc.archive.name}"},
    )
    monkeypatch.setenv("MDC_API_KEY", "key")
    monkeypatch.setattr(prepare.requests, "post", lambda *_, **__: session)
    yield session
    server.shutdown()
    server.server_close()


@pytest.mark.usefixtures("served_release")
def test_check_commonvoice_finds_all_clips_without_writing(dataset: Path) -> None:
    assert prepare.check_commonvoice(dataset, "swa").name == RELEASE
    assert not (dataset / "raw").exists()
    assert not any((dataset / "audio" / "swa" / "all").iterdir())


@pytest.mark.usefixtures("served_release")
def test_check_commonvoice_fails_on_missing_clips(dataset: Path) -> None:
    write_manifests(dataset, ["a", "gone"])
    with pytest.raises(prepare.MissingClipsError, match="1 of the 2 clips") as error:
        prepare.check_commonvoice(dataset, "swa")
    assert error.value.missing == ["gone"]


def test_check_commonvoice_reports_missing_terms_and_key(
    dataset: Path, served_release: SimpleNamespace, monkeypatch: pytest.MonkeyPatch
) -> None:
    served_release.status_code = 403
    with pytest.raises(PermissionError, match="Access denied"):
        prepare.check_commonvoice(dataset, "swa")
    monkeypatch.delenv("MDC_API_KEY")
    with pytest.raises(ValueError, match="MDC_API_KEY"):
        prepare.check_commonvoice(dataset, "swa")


def test_cli_commonvoice_checks_all_languages_and_fails_loudly(
    dataset: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    checked = []

    def check(_: Path, code: str) -> SimpleNamespace:
        checked.append(code)
        if code == "swa":
            raise prepare.MissingClipsError({"gone"}, 3, RELEASE)
        if code == "jpn":
            raise prepare.MissingClipsError({"lost"}, 4, f"release of {code}")
        return SimpleNamespace(name=f"release of {code}")

    monkeypatch.setattr(prepare, "inaccessible_releases", lambda _: [])
    monkeypatch.setattr(prepare, "check_commonvoice", check)
    with pytest.raises(SystemExit) as error:
        prepare.cli(["commonvoice", str(dataset), "--check-only"])
    assert checked == [lang.iso_639_3 for lang in commonvoice_languages()]
    message = str(error.value.code)
    assert "COMMON VOICE FAILED for 2 language(s)" in message
    assert "swa:" in message
    assert "gone" in message
    assert "jpn:" in message
    assert "lost" in message
    assert "tam: all clips found in release of tam" in capsys.readouterr().out


def test_inaccessible_releases_lists_releases_with_missing_terms(served_release: SimpleNamespace) -> None:
    assert prepare.inaccessible_releases(["swa"]) == []
    served_release.status_code = 403
    assert [release.name for release in prepare.inaccessible_releases(["swa"])] == [RELEASE]


@pytest.mark.parametrize("check_only", [False, True])
def test_cli_commonvoice_fails_fast_on_missing_terms(
    dataset: Path, mdc: FakeMDC, monkeypatch: pytest.MonkeyPatch, *, check_only: bool
) -> None:
    def deny(languages: list[str]) -> list[DatasetDetails]:
        assert languages == ["swa"]
        return [DatasetDetails(id="new", name=RELEASE, datasetUrl="https://mdc/datasets/new")]

    monkeypatch.setattr(prepare, "inaccessible_releases", deny)
    monkeypatch.setattr(prepare, "check_commonvoice", lambda *_: pytest.fail("checked before terms"))
    with pytest.raises(SystemExit) as error:
        prepare.cli(["commonvoice", str(dataset), "swa", *(["--check-only"] if check_only else [])])
    assert f"{RELEASE}: https://mdc/datasets/new" in str(error.value.code)
    assert mdc.downloads == 0


def prepare_all_clips(dataset: Path) -> None:
    for fileid in ["a", "b", "c"]:
        (dataset / "audio" / "swa" / "all" / f"{fileid}.wav").write_bytes(EXISTING)


def test_cli_commonvoice_skips_prepared_languages_without_api_key(
    dataset: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # A new release with unaccepted terms, or no API key, must not block a language that is already prepared
    prepare_all_clips(dataset)
    monkeypatch.delenv("MDC_API_KEY")
    monkeypatch.setattr(prepare, "latest_release", lambda _: pytest.fail("no request for a prepared language"))
    prepare.cli(["commonvoice", str(dataset), "swa"])
    assert "swa: all clips prepared" in capsys.readouterr().out


def test_cli_commonvoice_check_only_rechecks_prepared_languages(
    dataset: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    prepare_all_clips(dataset)
    terms, checked = [], []
    monkeypatch.setattr(prepare, "inaccessible_releases", lambda languages: terms.extend(languages) or [])
    monkeypatch.setattr(
        prepare, "check_commonvoice", lambda _, code: checked.append(code) or SimpleNamespace(name=RELEASE)
    )
    prepare.cli(["commonvoice", str(dataset), "swa", "--check-only"])
    assert terms == checked == ["swa"]
    assert f"swa: all clips found in {RELEASE}" in capsys.readouterr().out


def test_cli_commonvoice_requires_api_key_to_prepare(
    dataset: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("MDC_API_KEY")
    monkeypatch.setattr(prepare, "inaccessible_releases", lambda _: pytest.fail("no request without an API key"))
    with pytest.raises(SystemExit):
        prepare.cli(["commonvoice", str(dataset), "swa"])
    assert "Set `MDC_API_KEY`" in capsys.readouterr().err


@pytest.mark.usefixtures("served_release")
def test_cli_commonvoice_prepares_selected_languages(dataset: Path, mdc: FakeMDC) -> None:
    prepare.cli(["commonvoice", str(dataset), "swa"])
    assert mdc.downloads == 1
    assert len(list((dataset / "audio" / "swa" / "all").glob("*.wav"))) == 3


@pytest.fixture
def served_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Serve a tiny `discophon_data.tar.gz` over HTTP, and point `download_benchmark` to it.

    Yields:
        The served archive.

    """
    archive = tmp_path / "served" / "discophon_data.tar.gz"
    archive.parent.mkdir()
    with tarfile.open(archive, "w:gz") as tar:
        root = tarfile.TarInfo("discophon_data")
        root.type = tarfile.DIRTYPE
        tar.addfile(root)
        info = tarfile.TarInfo("discophon_data/manifest/speakers.jsonl")
        info.size = 2
        tar.addfile(info, io.BytesIO(b"{}"))
    server = ThreadingHTTPServer(("127.0.0.1", 0), functools.partial(RangeHandler, directory=archive.parent))
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setattr(RangeHandler, "requested_ranges", [])
    monkeypatch.setattr(prepare, "BENCHMARK_URL", f"http://127.0.0.1:{server.server_port}/{archive.name}")
    monkeypatch.setattr(prepare, "BENCHMARK_SHA256", hashlib.sha256(archive.read_bytes()).hexdigest())
    monkeypatch.setattr(prepare, "BENCHMARK_SIZE", archive.stat().st_size)
    yield archive
    server.shutdown()
    server.server_close()


@pytest.mark.usefixtures("served_benchmark")
def test_download_benchmark_extracts_archive(tmp_path: Path) -> None:
    prepare.cli(["download", str(tmp_path / "data")])
    assert (tmp_path / "data" / "manifest" / "speakers.jsonl").read_text() == "{}"
    assert not (tmp_path / "data" / "discophon_data.tar.gz").exists()


@pytest.mark.usefixtures("served_benchmark")
def test_download_benchmark_rejects_corrupted_archive(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(prepare, "BENCHMARK_SHA256", "0" * 64)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        prepare.download_benchmark(tmp_path / "data")
    assert not any((tmp_path / "data").iterdir())


def test_download_benchmark_resumes_interrupted_download(tmp_path: Path, served_benchmark: Path) -> None:
    half = served_benchmark.stat().st_size // 2
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "discophon_data.tar.gz").write_bytes(served_benchmark.read_bytes()[:half])
    prepare.download_benchmark(tmp_path / "data")
    assert RangeHandler.requested_ranges == [f"bytes={half}-"]
    assert (tmp_path / "data" / "manifest" / "speakers.jsonl").read_text() == "{}"
    assert not (tmp_path / "data" / "discophon_data.tar.gz").exists()


@pytest.mark.usefixtures("served_benchmark")
def test_download_benchmark_restarts_without_range_support(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(RangeHandler, "honor_range", False)
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "discophon_data.tar.gz").write_bytes(b"garbage")
    prepare.download_benchmark(tmp_path / "data")
    assert (tmp_path / "data" / "manifest" / "speakers.jsonl").read_text() == "{}"


@pytest.mark.usefixtures("served_benchmark")
def test_download_benchmark_keeps_archive_when_extraction_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(*_: object, **__: object) -> None:
        raise OSError("No space left on device")

    with monkeypatch.context() as patch:
        patch.setattr(tarfile.TarFile, "extract", fail)
        with pytest.raises(OSError, match="No space left"):
            prepare.download_benchmark(tmp_path / "data")
    assert (tmp_path / "data" / "discophon_data.tar.gz").is_file()
    prepare.download_benchmark(tmp_path / "data")  # the complete archive is extracted without downloading it again
    assert len(RangeHandler.requested_ranges) == 1
    assert (tmp_path / "data" / "manifest" / "speakers.jsonl").read_text() == "{}"
