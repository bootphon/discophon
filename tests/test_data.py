"""Tests for data loading/writing utilities and filename conventions."""

from decimal import Decimal
from itertools import pairwise
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

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
    read_submitted_units,
    textgrid_array_from_sequence,
    units_filename,
)
from discophon.languages import get_language

GERMAN = get_language("german")


def test_filename_helpers() -> None:
    assert units_filename(GERMAN, "dev") == "units-deu-dev.jsonl"
    assert alignment_filename(GERMAN, "test") == "alignment-deu-test.txt"
    assert item_filename(GERMAN, "dev", kind="triphone") == "triphone-deu-dev.item"
    assert manifest_filename(GERMAN, "train-1h") == "manifest-deu-train-1h.csv"


def test_textgrid_array_basic() -> None:
    entries = textgrid_array_from_sequence(["a", "a", "b"], step_in_ms=10)
    assert entries == [
        {"begin": 0.0, "end": 0.02, "label": "a"},
        {"begin": 0.02, "end": 0.03, "label": "b"},
    ]


@given(st.lists(st.sampled_from("abc"), min_size=1, max_size=30), st.integers(1, 40))
def test_textgrid_array_is_contiguous_and_covers_sequence(seq: list[str], step: int) -> None:
    entries = textgrid_array_from_sequence(seq, step_in_ms=step)
    # intervals are contiguous: each begins where the previous ended, starting at 0
    assert entries[0]["begin"] == pytest.approx(0.0)
    for prev, nxt in pairwise(entries):
        assert nxt["begin"] == pytest.approx(prev["end"])
    # consecutive labels differ (groupby collapsed the runs)
    labels = [e["label"] for e in entries]
    assert all(a != b for a, b in pairwise(labels))
    # total duration matches the number of tokens
    assert entries[-1]["end"] == pytest.approx(len(seq) * step / 1000)


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


def test_read_submitted_units_roundtrip(tmp_path: Path) -> None:
    path = tmp_path / "units.jsonl"
    path.write_text('{"file": "a", "units": [1, 2, 3]}\n{"file": "b", "units": [4, 5]}\n', encoding="utf-8")
    assert read_submitted_units(path) == {"a": [1, 2, 3], "b": [4, 5]}
