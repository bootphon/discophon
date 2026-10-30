"""Tests for data loading/writing utilities and filename conventions."""

import tempfile
from decimal import Decimal
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
    df_to_textgrids,
    item_filename,
    manifest_filename,
    num_invalid_rows,
    read_gold_annotations,
    read_rttm,
    read_scores,
    read_submitted_units,
    read_textgrid,
    rttm_to_textgrids,
    units_filename,
    write_textgrids,
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


def test_write_textgrids_merges_runs_of_tokens(tmp_path: Path) -> None:
    write_textgrids({"f": ["a", "a", "b"]}, tmp_path, tier_name="phones", step_in_ms=10)
    tier = read_textgrid(tmp_path / "f.TextGrid")["phones"]
    assert tier.select("text", "start", "end").rows() == [("a", 0.0, 0.02), ("b", 0.02, 0.03)]


@given(
    st.dictionaries(
        st.sampled_from(["f1", "f2", "f3"]),
        st.lists(
            st.sampled_from(["a", "b", "", '"', 'say ""hi""', "multi\nline", "ʈ͡ʂʰ", " c "]), min_size=1, max_size=30
        ),
        min_size=1,
    ),
    st.integers(1, 40),
)
def test_write_and_read_textgrids_round_trip(seqs: dict[str, list[str]], step: int) -> None:
    with tempfile.TemporaryDirectory() as outdir:
        write_textgrids(seqs, outdir, tier_name="phones", step_in_ms=step)
        tier = read_textgrid(outdir)["phones"]
    assert tier["fileid"].unique(maintain_order=True).to_list() == sorted(seqs)
    for (fileid,), df in tier.group_by("fileid"):
        # intervals are contiguous from 0, and consecutive labels differ (runs of tokens are merged)
        assert df["start"][0] == 0.0
        assert (df["start"][1:] == df["end"][:-1]).all()
        assert (df["text"][1:] != df["text"][:-1]).all()
        tokens = [
            text
            for text, start, end in df.select("text", "start", "end").rows()
            for _ in range(round((end - start) * 1000 / step))
        ]
        assert tokens == [token or "SIL" for token in seqs[str(fileid)]]


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


def test_write_textgrids_rejects_empty_sequences_with_their_files(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"1 files, such as \['b'\]"):
        write_textgrids({"a": ["x"], "b": []}, tmp_path, tier_name="phones", step_in_ms=10)
    assert not list(tmp_path.iterdir())  # nothing written


def test_write_and_read_textgrids(tmp_path: Path) -> None:
    write_textgrids({"f1": ["a", "a", ""], "f2": ["b"]}, tmp_path, tier_name="phones", step_in_ms=10)
    write_textgrids({"f1": [3, 4, 4], "f2": [5]}, tmp_path, tier_name="units", step_in_ms=10)  # added to the files
    assert sorted(p.name for p in tmp_path.iterdir()) == ["f1.TextGrid", "f2.TextGrid"]
    single = read_textgrid(tmp_path / "f1.TextGrid")
    assert set(single) == {"phones", "units"}
    assert single["phones"].to_dicts() == [
        {"text": "a", "start": 0.0, "end": 0.02, "fileid": "f1"},
        {"text": "SIL", "start": 0.02, "end": 0.03, "fileid": "f1"},  # empty labels are silences
    ]
    assert single["units"]["text"].to_list() == ["3", "4"]
    both = read_textgrid(tmp_path)
    assert both["phones"]["fileid"].to_list() == ["f1", "f1", "f2"]
    assert both["units"]["text"].to_list() == ["3", "4", "5"]


def test_read_textgrid_rejects_invalid_paths(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No TextGrid files"):
        read_textgrid(tmp_path)
    with pytest.raises(ValueError, match="neither a TextGrid file nor a directory"):
        read_textgrid(tmp_path / "missing.TextGrid")


def test_df_to_textgrids_sorts_intervals_and_adds_tiers(tmp_path: Path) -> None:
    df = pl.DataFrame(
        {"file": ["f1", "f2", "f1"], "begin": [0.5, 0.0, 0.0], "end": [1.0, 0.3, 0.5], "label": ["y", "z", "x"]}
    )
    df_to_textgrids(df, tmp_path, file_col="file", begin_col="begin", end_col="end", label_col="label", tier_name="a")
    df_to_textgrids(df, tmp_path, file_col="file", begin_col="begin", end_col="end", label_col="label", tier_name="b")
    tiers = read_textgrid(tmp_path / "f1.TextGrid")
    assert set(tiers) == {"a", "b"}
    assert tiers["a"].select("text", "start", "end").rows() == [("x", 0.0, 0.5), ("y", 0.5, 1.0)]
    assert read_textgrid(tmp_path / "f2.TextGrid")["a"]["text"].to_list() == ["z"]


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


def test_rttm_to_textgrids(tmp_path: Path) -> None:
    (tmp_path / "turns.rttm").write_text(RTTM, encoding="utf-8")
    rttm_to_textgrids(tmp_path / "turns.rttm", tmp_path / "grids", tier_name="speakers")
    tiers = read_textgrid(tmp_path / "grids")
    assert tiers["speakers"].select("fileid", "text", "start", "end").rows() == [
        ("f1", "spk1", 0.0, 0.5),
        ("f1", "spk2", 0.5, 0.75),
        ("f2", "SIL", 0.0, 0.1),  # gaps are filled with empty intervals
        ("f2", "spk1", 0.1, 1.1),
    ]


# Written by Praat 6, with trailing spaces, quotes escaped by doubling them, and a label on several lines
PRAAT_TEXTGRID = (
    'File type = "ooTextFile"\n'
    'Object class = "TextGrid"\n'
    "\n"
    "xmin = 0 \n"
    "xmax = 1 \n"
    "tiers? <exists> \n"
    "size = 2 \n"
    "item []: \n"
    "    item [1]:\n"
    '        class = "IntervalTier" \n'
    '        name = "phones ""ipa""" \n'
    "        xmin = 0 \n"
    "        xmax = 1 \n"
    "        intervals: size = 3 \n"
    "        intervals [1]:\n"
    "            xmin = 0 \n"
    "            xmax = 0.3 \n"
    '            text = "ʃ" \n'
    "        intervals [2]:\n"
    "            xmin = 0.3 \n"
    "            xmax = 0.6 \n"
    '            text = "say ""hi""\n'
    "xmin = 9\n"
    'second line" \n'
    "        intervals [3]:\n"
    "            xmin = 0.6 \n"
    "            xmax = 1 \n"
    '            text = "" \n'
    "    item [2]:\n"
    '        class = "TextTier" \n'
    '        name = "events" \n'
    "        xmin = 0 \n"
    "        xmax = 1 \n"
    "        points: size = 2 \n"
    "        points [1]:\n"
    "            number = 0.25 \n"
    '            mark = "click" \n'
    "        points [2]:\n"
    "            number = 0.75 \n"
    '            mark = "ɛ" \n'
)

# Written by the Montreal Forced Aligner, indented with tabs
MFA_TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 0.28
tiers? <exists>
size = 1
item []:
\titem [1]:
\t\tclass = "IntervalTier"
\t\tname = "words"
\t\txmin = 0.0
\t\txmax = 0.28
\t\tintervals: size = 2
\t\t\tintervals [1]:
\t\t\t\txmin = 0.0
\t\t\t\txmax = 0.160
\t\t\t\ttext = ""
\t\t\tintervals [2]:
\t\t\t\txmin = 0.160
\t\t\t\txmax = 0.280
\t\t\t\ttext = "and"
"""

ENCODINGS = {
    "utf-8": lambda text: text.encode("utf-8"),
    "utf-8-bom": lambda text: text.encode("utf-8-sig"),
    "utf-16-be": lambda text: b"\xfe\xff" + text.encode("utf-16-be"),
    "utf-16-le": lambda text: b"\xff\xfe" + text.encode("utf-16-le"),
}


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("encoding", ENCODINGS)
def test_read_textgrid_written_by_praat(tmp_path: Path, encoding: str, newline: str) -> None:
    (tmp_path / "f.TextGrid").write_bytes(ENCODINGS[encoding](PRAAT_TEXTGRID.replace("\n", newline)))
    tiers = read_textgrid(tmp_path / "f.TextGrid")
    assert list(tiers) == ['phones "ipa"', "events"]
    assert tiers['phones "ipa"'].rows() == [
        ("ʃ", 0.0, 0.3, "f"),
        (f'say "hi"{newline}xmin = 9{newline}second line', 0.3, 0.6, "f"),
        ("SIL", 0.6, 1.0, "f"),
    ]
    assert tiers["events"].columns == ["text", "pos", "fileid"]
    assert tiers["events"].rows() == [("click", 0.25, "f"), ("ɛ", 0.75, "f")]


def test_read_textgrid_directory_recursively_sorted_by_file_id(tmp_path: Path) -> None:
    (tmp_path / "z" / "chapter").mkdir(parents=True)
    (tmp_path / "z" / "chapter" / "a.TextGrid").write_text(MFA_TEXTGRID, encoding="utf-8")
    (tmp_path / "b.TextGrid").write_text(MFA_TEXTGRID, encoding="utf-8")
    words = read_textgrid(tmp_path)["words"]
    assert words.rows() == [
        ("SIL", 0.0, 0.16, "a"),
        ("and", 0.16, 0.28, "a"),
        ("SIL", 0.0, 0.16, "b"),
        ("and", 0.16, 0.28, "b"),
    ]


def test_read_textgrid_rejects_duplicate_file_ids(tmp_path: Path) -> None:
    for folder in ["x", "y"]:
        (tmp_path / folder).mkdir()
        (tmp_path / folder / "f.TextGrid").write_text(MFA_TEXTGRID, encoding="utf-8")
    with pytest.raises(ValueError, match=r"Duplicate TextGrid file ids in .*: 1, such as \['f'\]"):
        read_textgrid(tmp_path)


SHORT_TEXTGRID = (
    'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
    '0\n1\n<exists>\n1\n"IntervalTier"\n"a"\n0\n1\n1\n0\n1\n""\n'
)


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (b"", "not a TextGrid in the long text format"),
        (SHORT_TEXTGRID.encode(), "not a TextGrid in the long text format"),
        (b"ooBinaryFile\x08TextGrid\x00\x00", "Binary TextGrids are not supported"),
        (
            PRAAT_TEXTGRID.replace("ʃ", "é").encode("latin-1", errors="replace"),
            "neither encoded in UTF-8 nor in UTF-16",
        ),
        (PRAAT_TEXTGRID.replace('"events"', '"phones ""ipa"""').encode(), "Duplicate tier names"),
        (PRAAT_TEXTGRID.replace('"TextTier"', '"PitchTier"').encode(), "Unsupported TextGrid tiers"),
        (PRAAT_TEXTGRID.replace("size = 2 \nitem", "size = 3 \nitem").encode(), "Malformed TextGrids"),
        (PRAAT_TEXTGRID.replace("intervals: size = 3", "intervals: size = 4").encode(), "Malformed TextGrids"),
        (PRAAT_TEXTGRID.replace("xmax = 0.3 ", "xmax = abc").encode(), "Malformed TextGrids"),
        (PRAAT_TEXTGRID.replace('"click"', '"cl"ick"').encode(), "Malformed TextGrids"),
    ],
    ids=[
        "empty",
        "short",
        "binary",
        "latin-1",
        "duplicate tier",
        "pitch tier",
        "missing tier",
        "missing interval",
        "invalid time",
        "unescaped quote",
    ],
)
def test_read_textgrid_rejects_invalid_files(tmp_path: Path, content: bytes, match: str) -> None:
    (tmp_path / "f.TextGrid").write_bytes(content)
    with pytest.raises(ValueError, match=match):
        read_textgrid(tmp_path / "f.TextGrid")


def test_read_textgrid_rejects_every_truncation(tmp_path: Path) -> None:
    path = tmp_path / "f.TextGrid"
    path.write_text(PRAAT_TEXTGRID, encoding="utf-8")
    expected = read_textgrid(path)
    for size in range(len(PRAAT_TEXTGRID)):
        path.write_text(PRAAT_TEXTGRID[:size], encoding="utf-8")
        if size <= PRAAT_TEXTGRID.rindex('"'):  # only whitespace follows the last quote
            with pytest.raises(ValueError, match=r"long text format|Malformed"):
                read_textgrid(path)
        else:
            tiers = read_textgrid(path)
            assert list(tiers) == list(expected)
            assert all(tiers[name].equals(expected[name]) for name in tiers)


def test_read_textgrid_rejects_tier_with_different_kinds(tmp_path: Path) -> None:
    (tmp_path / "a.TextGrid").write_text(PRAAT_TEXTGRID, encoding="utf-8")
    (tmp_path / "b.TextGrid").write_text(MFA_TEXTGRID.replace('"words"', '"events"'), encoding="utf-8")
    with pytest.raises(ValueError, match="interval tiers in some TextGrids and point tiers in others"):
        read_textgrid(tmp_path)


def test_read_textgrid_without_tiers_or_entries(tmp_path: Path) -> None:
    (tmp_path / "absent.TextGrid").write_text(
        'File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = 0\nxmax = 1\ntiers? <absent>\n',
        encoding="utf-8",
    )
    assert read_textgrid(tmp_path / "absent.TextGrid") == {}
    (tmp_path / "empty.TextGrid").write_text(
        PRAAT_TEXTGRID[: PRAAT_TEXTGRID.index("        points: size")] + "        points: size = 0 \n",
        encoding="utf-8",
    )
    events = read_textgrid(tmp_path / "empty.TextGrid")["events"]
    assert events.is_empty()
    assert events.schema == {"text": pl.String, "pos": pl.Float64, "fileid": pl.String}


def test_write_textgrids_format(tmp_path: Path) -> None:
    df = pl.DataFrame({"file": ["f", "f"], "begin": [0.25, 0.5], "end": [0.5, 1.0], "label": ['say "hi"', "ʃ"]})
    df_to_textgrids(df, tmp_path, file_col="file", begin_col="begin", end_col="end", label_col="label", tier_name="a")
    write_textgrids({"f": [3, 3, 4]}, tmp_path, tier_name='b "c"', step_in_ms=500)
    assert (tmp_path / "f.TextGrid").read_text(encoding="utf-8") == (
        'File type = "ooTextFile"\n'
        'Object class = "TextGrid"\n'
        "\n"
        "xmin = 0.0\n"
        "xmax = 1.5\n"
        "tiers? <exists>\n"
        "size = 2\n"
        "item []:\n"
        "    item [1]:\n"
        '        class = "IntervalTier"\n'
        '        name = "a"\n'
        "        xmin = 0.0\n"
        "        xmax = 1.5\n"
        "        intervals: size = 4\n"
        "        intervals [1]:\n"
        "            xmin = 0.0\n"
        "            xmax = 0.25\n"
        '            text = ""\n'
        "        intervals [2]:\n"
        "            xmin = 0.25\n"
        "            xmax = 0.5\n"
        '            text = "say ""hi"""\n'
        "        intervals [3]:\n"
        "            xmin = 0.5\n"
        "            xmax = 1.0\n"
        '            text = "ʃ"\n'
        "        intervals [4]:\n"
        "            xmin = 1.0\n"
        "            xmax = 1.5\n"
        '            text = ""\n'
        "    item [2]:\n"
        '        class = "IntervalTier"\n'
        '        name = "b ""c"""\n'
        "        xmin = 0.0\n"
        "        xmax = 1.5\n"
        "        intervals: size = 2\n"
        "        intervals [1]:\n"
        "            xmin = 0.0\n"
        "            xmax = 1.0\n"
        '            text = "3"\n'
        "        intervals [2]:\n"
        "            xmin = 1.0\n"
        "            xmax = 1.5\n"
        '            text = "4"\n'
    )


def test_write_textgrids_keeps_other_tiers_and_replaces_same_name(tmp_path: Path) -> None:
    (tmp_path / "f.TextGrid").write_text(PRAAT_TEXTGRID, encoding="utf-8")
    original = read_textgrid(tmp_path / "f.TextGrid")
    write_textgrids({"f": [1, 2]}, tmp_path, tier_name="units", step_in_ms=500)
    write_textgrids({"f": [5]}, tmp_path, tier_name="units", step_in_ms=500)
    tiers = read_textgrid(tmp_path / "f.TextGrid")
    assert list(tiers) == ['phones "ipa"', "events", "units"]
    assert tiers['phones "ipa"'].equals(original['phones "ipa"'])
    assert tiers["events"].equals(original["events"])
    assert tiers["units"].rows() == [("5", 0.0, 0.5, "f"), ("SIL", 0.5, 1.0, "f")]  # extended to the end of the file
    assert [p.name for p in tmp_path.iterdir()] == ["f.TextGrid"]


def test_df_to_textgrids_rounds_times_to_the_microsecond(tmp_path: Path) -> None:
    df = pl.DataFrame({"file": ["f", "f"], "begin": [0.0, 0.3], "end": [0.1 + 0.2, 0.5], "label": ["x", "y"]})
    df_to_textgrids(df, tmp_path, file_col="file", begin_col="begin", end_col="end", label_col="label", tier_name="a")
    tier = read_textgrid(tmp_path / "f.TextGrid")["a"]
    assert tier.select("text", "start", "end").rows() == [("x", 0.0, 0.3), ("y", 0.3, 0.5)]  # no overlap, nor gap


@pytest.mark.parametrize(
    ("begin", "end", "label"),
    [
        ([0.0, 0.4], [0.5, 1.0], ["x", "y"]),  # overlap
        ([0.0, 0.5], [0.5, 0.5], ["x", "y"]),  # empty interval
        ([0.0, 0.5], [0.5, 0.4], ["x", "y"]),  # negative duration
        ([-0.5, 0.5], [0.5, 1.0], ["x", "y"]),  # before 0
        ([0.0, 0.5], [0.5, 1.0], ["x", None]),
        ([0.0, None], [0.5, 1.0], ["x", "y"]),
    ],
)
def test_df_to_textgrids_rejects_invalid_intervals(
    tmp_path: Path, begin: list[float | None], end: list[float], label: list[str | None]
) -> None:
    df = pl.DataFrame({"file": ["f", "f"], "begin": begin, "end": end, "label": label})
    with pytest.raises(
        ValueError, match=r"null|must not start before 0, must last more than 0 s, and must not overlap"
    ):
        df_to_textgrids(
            df, tmp_path, file_col="file", begin_col="begin", end_col="end", label_col="label", tier_name="a"
        )
    assert not (tmp_path / "f.TextGrid").exists()


def test_df_to_textgrids_with_no_intervals(tmp_path: Path) -> None:
    df = pl.DataFrame(schema={"file": pl.String, "begin": pl.Float64, "end": pl.Float64, "label": pl.String})
    df_to_textgrids(
        df, tmp_path / "out", file_col="file", begin_col="begin", end_col="end", label_col="label", tier_name="a"
    )
    assert not list((tmp_path / "out").iterdir())
