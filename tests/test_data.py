"""Tests for data loading/writing utilities and filename conventions."""

from decimal import Decimal
from pathlib import Path

import polars as pl
import pytest

from discophon.data import (
    FILE,
    OFFSET,
    ONSET,
    PHONE,
    alignment_filename,
    decimal_series_is_integer,
    item_filename,
    manifest_filename,
    num_invalid_rows,
    read_gold_annotations,
    read_rttm,
    read_scores,
    read_submitted_units,
    units_filename,
)
from discophon.languages import get_language

GERMAN = get_language("german")


@pytest.mark.parametrize("column", range(4))
def test_read_gold_annotations_rejects_null_fields(tmp_path: Path, column: int) -> None:
    fields = ["f", "0.02", "0.04", "b"]
    fields[column] = ""
    path = tmp_path / "alignment.txt"
    path.write_text("#file onset offset #phone\nf 0 0.02 a\n" + " ".join(fields) + "\n")
    with pytest.raises(ValueError, match="must not be null"):
        read_gold_annotations(path)


def test_filename_helpers() -> None:
    assert units_filename(GERMAN, "dev") == "units-deu-dev.jsonl"
    assert alignment_filename(GERMAN, "test") == "alignment-deu-test.txt"
    assert item_filename(GERMAN, "dev", kind="triphone") == "triphone-deu-dev.item"
    assert manifest_filename(GERMAN, "train-1h") == "manifest-deu-train-1h.csv"


def test_num_invalid_rows_accepts_contiguous_alignment() -> None:
    df = pl.DataFrame(
        {
            FILE: ["f", "f"],
            ONSET: [Decimal(0), Decimal("0.1")],
            OFFSET: [Decimal("0.1"), Decimal("0.2")],
            PHONE: ["a", "b"],
        }
    )
    assert num_invalid_rows(df, step_in_ms=10) == 0


def test_num_invalid_rows_flags_gap() -> None:
    df = pl.DataFrame(
        {
            FILE: ["f", "f"],
            ONSET: [Decimal(0), Decimal("0.2")],  # gap: should start at 0.1
            OFFSET: [Decimal("0.1"), Decimal("0.3")],
            PHONE: ["a", "b"],
        }
    )
    assert num_invalid_rows(df, step_in_ms=10) == 1


def test_num_invalid_rows_flags_nonzero_start() -> None:
    df = pl.DataFrame({FILE: ["f"], ONSET: [Decimal("0.05")], OFFSET: [Decimal("0.1")], PHONE: ["a"]})
    assert num_invalid_rows(df, step_in_ms=10) == 1


def test_decimal_series_is_integer() -> None:
    assert decimal_series_is_integer(pl.Series([Decimal("2.0"), Decimal("3.000")]))
    assert not decimal_series_is_integer(pl.Series([Decimal("2.5"), Decimal("3.0")]))


def test_read_gold_annotations_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "alignment.txt"
    path.write_text("#file onset offset #phone\nf 0 0.02 a\nf 0.02 0.05 b\n", encoding="utf-8")
    annotations = read_gold_annotations(path)
    # "a" spans 0-0.02 (2 frames), "b" spans 0.02-0.05 (3 frames) at 10 ms each
    assert annotations == {"f": ["a", "a", "b", "b", "b"]}


def test_read_gold_annotations_rejects_unaligned(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("#file onset offset #phone\nf 0 0.02 a\nf 0.03 0.05 b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid annotations"):
        read_gold_annotations(path)


def test_read_gold_annotations_rejects_zero_duration_first_entry(tmp_path: Path) -> None:
    path = tmp_path / "bad.txt"
    path.write_text("#file onset offset #phone\nf 0.00 0.00 a\nf 0.00 0.02 b\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid annotations"):
        read_gold_annotations(path)


def test_read_submitted_units_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "units.jsonl"
    path.write_text('{"file": "a", "units": [1, 2, 3]}\n{"file": "b", "units": [4, 5]}\n', encoding="utf-8")
    assert read_submitted_units(path) == {"a": [1, 2, 3], "b": [4, 5]}


def write_scores(path: Path, rows: list[tuple[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(
        [{"language": "deu", "split": "test", "metric": metric, "score": score} for metric, score in rows],
        orient="row",
    ).write_ndjson(path)
    (path.parents[3] / "info.json").write_text("{}")


def test_read_scores_averages_abx_and_keeps_individual_scores(tmp_path: Path) -> None:
    write_scores(
        tmp_path / "my-model" / "zero-shot" / "continuous" / "6" / "scores.jsonl",
        [
            ("triphone_abx_continuous_within_speaker", 0.1),
            ("triphone_abx_continuous_across_speaker", 0.2),
            ("phoneme_abx_continuous_within_speaker_within_context", 0.3),
            ("phoneme_abx_continuous_across_speaker_within_context", 0.4),
            ("phoneme_abx_continuous_within_speaker_any_context", 0.5),
            ("phoneme_abx_continuous_across_speaker_any_context", 0.6),
        ],
    )
    write_scores(tmp_path / "my-model" / "ft-eng-1h" / "many_to_one" / "12" / "scores.jsonl", [("per", 0.7)])
    df = read_scores(tmp_path)
    scores = dict(df.select("metric", "score").iter_rows())
    assert len(scores) == len(df) == 10
    assert scores["triphone_abx_continuous"] == pytest.approx(0.15)
    assert scores["phoneme_abx_continuous_within_context"] == pytest.approx(0.35)
    assert scores["phoneme_abx_continuous_any_context"] == pytest.approx(0.55)
    assert scores["phoneme_abx_continuous_across_speaker_any_context"] == 0.6  # ruff: ignore[magic-value-comparison]
    per = df.filter(metric="per").row(0, named=True)
    assert (per["model"], per["folder"], per["ft_lang"], per["duration"], per["layer"]) == (
        "my-model",
        "many_to_one",
        "eng",
        "1h",
        12,
    )
    assert per["language_split"] == GERMAN.split
    assert df.filter(metric="triphone_abx_continuous")["ft_lang"].item() is None
    assert read_scores(tmp_path / "my-model").equals(df)


def test_read_scores_rejects_missing_speaker_condition(tmp_path: Path) -> None:
    write_scores(
        tmp_path / "m" / "zero-shot" / "continuous" / "1" / "scores.jsonl",
        [("triphone_abx_continuous_within_speaker", 0.1)],
    )
    with pytest.raises(ValueError, match="within speaker without across speaker"):
        read_scores(tmp_path)


def test_read_scores_rejects_duplicates_and_empty(tmp_path: Path) -> None:
    write_scores(tmp_path / "m" / "zero-shot" / "many_to_one" / "1" / "scores.jsonl", [("per", 0.1), ("per", 0.2)])
    with pytest.raises(ValueError, match="Duplicate"):
        read_scores(tmp_path)
    with pytest.raises(ValueError, match="No scores"):
        read_scores(tmp_path / "missing")


def test_read_submitted_units_rejects_duplicate_files(tmp_path: Path) -> None:
    path = tmp_path / "units.jsonl"
    path.write_text('{"file": "a", "units": [1]}\n{"file": "b", "units": [2]}\n{"file": "a", "units": [3]}\n')
    with pytest.raises(ValueError, match=r"Duplicate files in .*: 1 files, such as \['a'\]"):
        read_submitted_units(path)


@pytest.mark.parametrize(
    "line", ['{"file": "b"}', '{"file": "b", "units": null}', '{"file": "b", "units": [1, null]}']
)
def test_read_submitted_units_rejects_missing_units(tmp_path: Path, line: str) -> None:
    path = tmp_path / "units.jsonl"
    path.write_text(f'{{"file": "a", "units": [1]}}\n{line}\n')
    with pytest.raises(ValueError, match=r"Missing or null units in .*: 1 files, such as \['b'\]"):
        read_submitted_units(path)


RTTM = """SPEAKER f1 1 0.00 0.50 <NA> <NA> spk1 <NA> <NA>
SPEAKER f1 1 0.50 0.25 <NA> <NA> spk2 <NA> <NA>
SPEAKER f2 1 0.10 1.00 <NA> <NA> spk1 <NA> <NA>
"""


def test_read_rttm(tmp_path: Path) -> None:
    (tmp_path / "turns.rttm").write_text(RTTM, encoding="utf-8")
    df = read_rttm(tmp_path / "turns.rttm")
    assert df.columns[:5] == ["Type", "File ID", "Channel ID", "Turn Onset", "Turn Duration"]
    assert df.select("File ID", "Turn Onset", "Turn Duration", "Speaker Name").rows() == [
        ("f1", 0.0, 0.5, "spk1"),
        ("f1", 0.5, 0.25, "spk2"),
        ("f2", 0.1, 1.0, "spk1"),
    ]
    assert df["Orthography Field"].null_count() == len(df)
