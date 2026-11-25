"""Tests for reading and writing Praat TextGrids, and for the TextGrid export entry point."""

import contextlib
import json
import tempfile
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st

from discophon.data import alignment_filename, units_filename
from discophon.languages import get_language
from discophon.textgrid import cli, df_to_textgrids, read_textgrid, rttm_to_textgrids, write_textgrids

from .test_data import RTTM

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

# The same TextGrid, saved by Praat as a short text file, with a comment
SHORT_TEXTGRID = (
    'File type = "ooTextFile"\n'
    'Object class = "TextGrid"\n'
    "\n"
    "0\n1\n<exists>\n2\n"
    '"IntervalTier"\n"phones ""ipa"""\n0\n1\n3\n'
    '0\n0.3\n"ʃ"\n'
    '0.3\n0.6\n"say ""hi""\nxmin = 9\nsecond line"\n'
    '0.6\n1\n""\n'
    '"TextTier"\n"events"\n0\n1\n2 ! number of points\n'
    '0.25\n"click"\n'
    '0.75\n"ɛ"\n'
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

# Written by praat-textgrids: no gap filling, point tiers named PointTier with keys xpos and text
PRAAT_TEXTGRIDS_TEXTGRID = """File type = "ooTextFile"
Object class = "TextGrid"

xmin = 0.0
xmax = 2.0
tiers? <exists>
size = 2
item []:
    item [1]:
        class = "IntervalTier"
        name = "speech"
        xmin = 0.0
        xmax = 2.0
        intervals: size = 2
            intervals [1]:
                xmin = 0.5
                xmax = 1.0
                text = "a"
            intervals [2]:
                xmin = 1.5
                xmax = 1.9889999999999999
                text = "b"
    item [2]:
        class = "PointTier"
        name = "events"
        xmin = 0.0
        xmax = 2.0
        points: size = 1
            points [1]:
                xpos = 1.25
                text = "x"
"""

ENCODINGS = {
    "utf-8": lambda text: text.encode("utf-8"),
    "utf-8-bom": lambda text: text.encode("utf-8-sig"),
    "utf-16-be": lambda text: b"\xfe\xff" + text.encode("utf-16-be"),
    "utf-16-le": lambda text: b"\xff\xfe" + text.encode("utf-16-le"),
}


def test_write_textgrids_merges_runs_of_tokens(tmp_path: Path) -> None:
    write_textgrids({"f": ["a", "a", "b"]}, tmp_path, tier_name="phones", step_in_ms=10)
    tier = read_textgrid(tmp_path / "f.TextGrid")["phones"]
    assert tier.select("text", "start", "end").rows() == [("a", 0.0, 0.02), ("b", 0.02, 0.03)]


LABELS = ["a", "b", "", '"', 'say ""hi""', "multi\nline", "ʈ͡ʂʰ", " c ", "!", "[1]", "<exists>", "0.5", "-"]


@given(
    st.dictionaries(
        st.sampled_from(["f1", "f2", "f3"]), st.lists(st.sampled_from(LABELS), min_size=1, max_size=30), min_size=1
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


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
@pytest.mark.parametrize("encoding", ENCODINGS)
@pytest.mark.parametrize("content", [PRAAT_TEXTGRID, SHORT_TEXTGRID], ids=["long", "short"])
def test_read_textgrid_written_by_praat(tmp_path: Path, content: str, encoding: str, newline: str) -> None:
    (tmp_path / "f.TextGrid").write_bytes(ENCODINGS[encoding](content.replace("\n", newline)))
    tiers = read_textgrid(tmp_path / "f.TextGrid")
    assert list(tiers) == ['phones "ipa"', "events"]
    assert tiers['phones "ipa"'].rows() == [
        ("ʃ", 0.0, 0.3, "f"),
        (f'say "hi"{newline}xmin = 9{newline}second line', 0.3, 0.6, "f"),
        ("SIL", 0.6, 1.0, "f"),
    ]
    assert tiers["events"].columns == ["text", "pos", "fileid"]
    assert tiers["events"].rows() == [("click", 0.25, "f"), ("ɛ", 0.75, "f")]


def test_read_textgrid_written_by_praat_textgrids(tmp_path: Path) -> None:
    (tmp_path / "f.TextGrid").write_text(PRAAT_TEXTGRIDS_TEXTGRID, encoding="utf-8")
    tiers = read_textgrid(tmp_path / "f.TextGrid")
    assert tiers["speech"].rows() == [("a", 0.5, 1.0, "f"), ("b", 1.5, 1.9889999999999999, "f")]  # times as written
    assert tiers["events"].rows() == [("x", 1.25, "f")]
    write_textgrids({"f": [1]}, tmp_path, tier_name="units", step_in_ms=20)  # rewritten as Praat would
    tiers = read_textgrid(tmp_path / "f.TextGrid")
    assert tiers["speech"].rows() == [
        ("SIL", 0.0, 0.5, "f"),
        ("a", 0.5, 1.0, "f"),
        ("SIL", 1.0, 1.5, "f"),
        ("b", 1.5, 1.9889999999999999, "f"),
        ("SIL", 1.9889999999999999, 2.0, "f"),
    ]
    assert tiers["events"].rows() == [("x", 1.25, "f")]
    assert 'class = "TextTier"' in (tmp_path / "f.TextGrid").read_text(encoding="utf-8")


def test_read_textgrid_replaces_empty_labels_with_empty_label(tmp_path: Path) -> None:
    (tmp_path / "f.TextGrid").write_text(PRAAT_TEXTGRID, encoding="utf-8")
    default = read_textgrid(tmp_path / "f.TextGrid")
    assert any("SIL" in tier["text"] for tier in default.values())
    for empty_label in ["SIL", "", "<sil>"]:
        tiers = read_textgrid(tmp_path / "f.TextGrid", empty_label=empty_label)
        assert tiers.keys() == default.keys()
        for name, tier in tiers.items():
            expected = default[name].with_columns(pl.col("text").replace("SIL", empty_label))
            assert tier.equals(expected)


def test_read_textgrid_fills_gaps_as_praat_writes_them(tmp_path: Path) -> None:
    (tmp_path / "f.TextGrid").write_text(PRAAT_TEXTGRIDS_TEXTGRID, encoding="utf-8")
    tiers = read_textgrid(tmp_path / "f.TextGrid", fill_gaps=True, empty_label="")
    assert tiers["speech"].rows() == [
        ("", 0.0, 0.5, "f"),
        ("a", 0.5, 1.0, "f"),
        ("", 1.0, 1.5, "f"),
        ("b", 1.5, 1.9889999999999999, "f"),
        ("", 1.9889999999999999, 2.0, "f"),
    ]
    assert tiers["events"].rows() == [("x", 1.25, "f")]  # point tiers have no gaps
    assert tiers["speech"].schema == {"text": pl.String, "start": pl.Float64, "end": pl.Float64, "fileid": pl.String}
    write_textgrids({"f": [1]}, tmp_path, tier_name="units", step_in_ms=20)  # rewritten as Praat would
    rewritten = read_textgrid(tmp_path / "f.TextGrid")
    assert read_textgrid(tmp_path / "f.TextGrid", fill_gaps=True)["speech"].equals(rewritten["speech"])
    assert rewritten["speech"].equals(tiers["speech"].with_columns(pl.col("text").replace("", "SIL")))


def test_read_textgrid_fills_gaps_of_files_without_gaps_or_intervals(tmp_path: Path) -> None:
    (tmp_path / "f.TextGrid").write_text(PRAAT_TEXTGRID, encoding="utf-8")
    original = read_textgrid(tmp_path / "f.TextGrid")
    filled = read_textgrid(tmp_path / "f.TextGrid", fill_gaps=True)
    assert filled.keys() == original.keys()
    assert all(filled[name].equals(original[name]) for name in original)
    empty = PRAAT_TEXTGRIDS_TEXTGRID.replace("intervals: size = 2", "intervals: size = 0")
    empty = empty[: empty.index("            intervals [1]")] + empty[empty.index("    item [2]") :]
    (tmp_path / "f.TextGrid").write_text(empty, encoding="utf-8")
    assert read_textgrid(tmp_path / "f.TextGrid", fill_gaps=True)["speech"].rows() == [("SIL", 0.0, 2.0, "f")]


@pytest.mark.parametrize(
    ("intervals", "match"),
    [
        ('0.5 1.0 "a" 0.8 1.2 "b"', "must not overlap"),
        ('1.0 1.5 "a" 0.2 0.5 "b"', "must not overlap"),
        ('0.5 0.5 "a" 1.0 1.5 "b"', "more than 0 s"),
        ('-1.0 0.5 "a" 1.0 1.5 "b"', "not start before"),
    ],
)
def test_read_textgrid_rejects_gaps_that_cannot_be_filled(tmp_path: Path, intervals: str, match: str) -> None:
    content = f'"ooTextFile" "TextGrid" 0 2 <exists> 1 "IntervalTier" "speech" 0 2 2 {intervals}\n'
    (tmp_path / "f.TextGrid").write_text(content, encoding="utf-8")
    read_textgrid(tmp_path / "f.TextGrid")  # Read as written without filling the gaps
    with pytest.raises(ValueError, match=f"Cannot fill the gaps of tier 'speech' of .*f.TextGrid, .*{match}"):
        read_textgrid(tmp_path / "f.TextGrid", fill_gaps=True)


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


@pytest.mark.parametrize(
    ("content", "match"),
    [
        (b"", "not a TextGrid in the long or short text format"),
        (b'File type = "ooTextFile"\nObject class = "Sound"\n', "not a TextGrid in the long or short text format"),
        (b"ooBinaryFile\x08TextGrid\x00\x00", "Binary TextGrids are not supported"),
        (
            PRAAT_TEXTGRID.replace("ʃ", "é").encode("latin-1", errors="replace"),
            "neither encoded in UTF-8 nor in UTF-16",
        ),
        (PRAAT_TEXTGRID.replace('"events"', '"phones ""ipa"""').encode(), "duplicate tier name"),
        (PRAAT_TEXTGRID.replace('"TextTier"', '"PitchTier"').encode(), "unsupported tier class 'PitchTier'"),
        (PRAAT_TEXTGRID.replace("<exists>", "<maybe>").encode(), "expected <exists> or <absent>, got '<maybe>'"),
        (PRAAT_TEXTGRID.replace("size = 2 \nitem", "size = 3 \nitem").encode(), "unexpected end of file"),
        (PRAAT_TEXTGRID.replace("size = 2 \nitem", "size = 1 \nitem").encode(), "more entries than declared"),
        (PRAAT_TEXTGRID.replace("size = 2 \nitem", "size = -1 \nitem").encode(), "negative size"),
        (PRAAT_TEXTGRID.replace("intervals: size = 3", "intervals: size = 4").encode(), "Invalid TextGrid"),
        (PRAAT_TEXTGRID.replace("intervals: size = 3", "intervals: size = 2").encode(), "Invalid TextGrid"),
        (PRAAT_TEXTGRID.replace("xmax = 0.3 ", "xmax = abc").encode(), "Invalid TextGrid"),
        (PRAAT_TEXTGRID.replace("xmax = 0.3 ", "xmax = 0.3e").encode(), "could not convert string to float"),
        (PRAAT_TEXTGRID.replace("xmax = 0.3 ", "xmax = 1e999").encode(), "times must be finite"),
        (PRAAT_TEXTGRID.replace('"ʃ"', "0.1").encode(), "expected a string, got '0.1'"),
        (PRAAT_TEXTGRID.replace('"click"', '"cl"ick"').encode(), "Invalid TextGrid"),
        (PRAAT_TEXTGRID.replace('"click"', '"click').encode(), "Invalid TextGrid"),
        (PRAAT_TEXTGRID.replace("item [1]", "item [1").encode(), "Invalid TextGrid"),
        (PRAAT_TEXTGRID + "0.5\n", "more entries than declared"),
    ],
    ids=[
        "empty",
        "sound",
        "binary",
        "latin-1",
        "duplicate tier",
        "pitch tier",
        "unknown flag",
        "missing tier",
        "extra tier",
        "negative size",
        "missing interval",
        "extra interval",
        "missing time",
        "invalid time",
        "infinite time",
        "number as label",
        "unescaped quote",
        "unclosed quote",
        "unclosed bracket",
        "trailing number",
    ],
)
def test_read_textgrid_rejects_invalid_files(tmp_path: Path, content: bytes | str, match: str) -> None:
    (tmp_path / "f.TextGrid").write_bytes(content if isinstance(content, bytes) else content.encode())
    with pytest.raises(ValueError, match=match):
        read_textgrid(tmp_path / "f.TextGrid")


@pytest.mark.parametrize("content", [PRAAT_TEXTGRID, SHORT_TEXTGRID], ids=["long", "short"])
def test_read_textgrid_rejects_every_truncation(tmp_path: Path, content: str) -> None:
    path = tmp_path / "f.TextGrid"
    path.write_text(content, encoding="utf-8")
    expected = read_textgrid(path)
    for size in range(len(content)):
        path.write_text(content[:size], encoding="utf-8")
        if size <= content.rindex('"'):  # only whitespace follows the last quote
            with pytest.raises(ValueError, match="Invalid TextGrid"):
                read_textgrid(path)
        else:
            tiers = read_textgrid(path)
            assert list(tiers) == list(expected)
            assert all(tiers[name].equals(expected[name]) for name in tiers)


@given(st.integers(0, len(PRAAT_TEXTGRID)), st.integers(0, 20), st.text(alphabet='"<>[]!.-+0123456789e \n'))
def test_read_textgrid_fails_only_with_value_errors(start: int, length: int, replacement: str) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        path = Path(tmpdir) / "f.TextGrid"
        path.write_text(PRAAT_TEXTGRID[:start] + replacement + PRAAT_TEXTGRID[start + length :], encoding="utf-8")
        with contextlib.suppress(ValueError):  # Any other exception is a bug
            read_textgrid(path)


def test_read_textgrid_rejects_tier_with_different_kinds(tmp_path: Path) -> None:
    (tmp_path / "a.TextGrid").write_text(PRAAT_TEXTGRID, encoding="utf-8")
    (tmp_path / "b.TextGrid").write_text(MFA_TEXTGRID.replace('"words"', '"events"'), encoding="utf-8")
    with pytest.raises(ValueError, match="interval tier in some TextGrids and a point tier in others"):
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


def test_write_textgrids_extends_existing_files_from_0_to_their_end(tmp_path: Path) -> None:
    content = PRAAT_TEXTGRID.replace("xmin = 0 \nxmax = 1 \n", "xmin = 0.1 \nxmax = 2 \n").replace(
        "xmin = 0 \n            xmax = 0.3", "xmin = 0.1 \n            xmax = 0.3"
    )
    (tmp_path / "f.TextGrid").write_text(content, encoding="utf-8")
    write_textgrids({"f": [1]}, tmp_path, tier_name="units", step_in_ms=500)
    tiers = read_textgrid(tmp_path / "f.TextGrid")
    assert tiers["units"].rows() == [("1", 0.0, 0.5, "f"), ("SIL", 0.5, 2.0, "f")]
    assert tiers['phones "ipa"'].select("text", "start", "end").rows() == [
        ("SIL", 0.0, 0.1),
        ("ʃ", 0.1, 0.3),
        ('say "hi"\nxmin = 9\nsecond line', 0.3, 0.6),
        ("SIL", 0.6, 1.0),
        ("SIL", 1.0, 2.0),
    ]
    assert tiers["events"]["pos"].to_list() == [0.25, 0.75]


@pytest.mark.parametrize(
    ("old", "new", "match"),
    [
        ("xmin = 0.3 \n", "xmin = 0.2 \n", "'phones \"ipa\"': invalid interval"),
        ("number = 0.25", "number = -0.25", "'events': points must not be before the TextGrid"),
    ],
    ids=["overlapping intervals", "point before start"],
)
def test_write_textgrids_refuses_to_rewrite_invalid_files(tmp_path: Path, old: str, new: str, match: str) -> None:
    invalid = PRAAT_TEXTGRID.replace(old, new)
    (tmp_path / "bad.TextGrid").write_text(invalid, encoding="utf-8")
    with pytest.raises(ValueError, match=rf"Cannot write .*bad.TextGrid, tier {match}"):
        write_textgrids({"good": [1], "bad": [2]}, tmp_path, tier_name="units", step_in_ms=10)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bad.TextGrid"]  # all or nothing
    assert (tmp_path / "bad.TextGrid").read_text(encoding="utf-8") == invalid


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
        ([0.0, 0.5], [0.5, float("nan")], ["x", "y"]),
        ([0.0, 0.5], [0.5, float("inf")], ["x", "y"]),
        ([0.0, 0.5], [0.5, 1.0], ["x", None]),
        ([0.0, None], [0.5, 1.0], ["x", "y"]),
    ],
)
def test_df_to_textgrids_rejects_invalid_intervals(
    tmp_path: Path, begin: list[float | None], end: list[float], label: list[str | None]
) -> None:
    df = pl.DataFrame({"file": ["f", "f"], "begin": begin, "end": end, "label": label})
    with pytest.raises(ValueError, match=r"null|must last more than 0 s, must not overlap, and must not start before"):
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


def test_cli_exports_phones_units_and_predictions(tmp_path: Path) -> None:
    language = get_language("deu")
    (tmp_path / "data" / "alignment").mkdir(parents=True)
    (tmp_path / "data" / "alignment" / alignment_filename(language, "dev")).write_text(
        "#file onset offset #phone\nf 0 0.04 a\nf 0.04 0.08 b\n", encoding="utf-8"
    )
    (tmp_path / "units").mkdir()
    for split in ["dev", "train-1h"]:  # train splits are skipped
        (tmp_path / "units" / units_filename(language, split)).write_text(
            json.dumps({"file": "f", "units": [7, 7, 3, 3]}) + "\n", encoding="utf-8"
        )
    cli([str(tmp_path / "data"), str(tmp_path / "units"), str(tmp_path / "out")])
    assert [p.relative_to(tmp_path / "out").as_posix() for p in (tmp_path / "out").rglob("*.TextGrid")] == [
        "deu/dev/f.TextGrid"
    ]
    tiers = read_textgrid(tmp_path / "out" / "deu" / "dev")
    assert list(tiers) == ["phones", "units", "predictions"]
    assert tiers["phones"].select("text", "start", "end").rows() == [("a", 0.0, 0.04), ("b", 0.04, 0.08)]
    assert tiers["units"].select("text", "start", "end").rows() == [("7", 0.0, 0.04), ("3", 0.04, 0.08)]
    assert tiers["predictions"].select("text", "start", "end").rows() == [("a", 0.0, 0.04), ("b", 0.04, 0.08)]
