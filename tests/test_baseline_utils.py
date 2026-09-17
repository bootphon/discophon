import json
from pathlib import Path

from discophon.baselines.utils import read_completed_fileids


def test_read_completed_fileids_supports_resuming_and_deduplicates(tmp_path: Path) -> None:
    output = tmp_path / "units.jsonl"
    entries = [{"file": "a", "units": [1]}, {"file": "a", "units": [1]}, {"file": "b", "units": [2]}]
    output.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")
    assert read_completed_fileids(output) == {"a", "b"}


def test_read_completed_fileids_accepts_missing_output(tmp_path: Path) -> None:
    assert read_completed_fileids(tmp_path / "missing.jsonl") == set()
