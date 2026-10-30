"""Read and write Praat TextGrids."""

import argparse
import itertools
import math
from pathlib import Path
from typing import NamedTuple

import polars as pl
from tqdm import tqdm

from discophon.benchmark import EVALUATED_SPLITS, available_languages_and_splits_for_units
from discophon.data import (
    DEFAULT_N_UNITS,
    STEP_PHONES,
    STEP_UNITS,
    Phones,
    Units,
    alignment_filename,
    read_gold_annotations,
    read_rttm,
    read_submitted_units,
    units_filename,
)
from discophon.evaluate import cooccurrence_matrix, phone_assignments

__all__ = ["df_to_textgrids", "read_textgrid", "rttm_to_textgrids", "write_textgrids"]

INTERVAL_TIER, POINT_TIER = "IntervalTier", "TextTier"
SCHEMAS = {
    INTERVAL_TIER: {"text": pl.String, "start": pl.Float64, "end": pl.Float64},
    POINT_TIER: {"text": pl.String, "pos": pl.Float64},
}

TOKEN = r'"[^"]*(?:""[^"]*)*"|<\w+>|[-+]?\.?\d[\d.eE+-]*|\[[^\]"\n]*\]|![^\n]*|["<>\[\]!.+-]'  # ruff: ignore[hardcoded-password-string]
"""A token of a TextGrid in the long or short text format: a string (where quotes are escaped by doubling them),
a flag, a number, an index in brackets, or a comment. Like Praat, the reader keeps only strings, flags, and numbers,
and ignores everything else (the keys of the long format). A character that starts no valid token, such as the
opening quote of a string that is never closed, is a token on its own, so that the TextGrid is invalid."""

HEADER = 'File type = "ooTextFile"\nObject class = "TextGrid"\n\nxmin = {}\nxmax = {}\ntiers? <exists>\nsize = {}\n'
TIER = '    item [{}]:\n        class = "{}"\n        name = "{}"\n        xmin = {}\n        xmax = {}\n'
INTERVAL = '        intervals [{}]:\n            xmin = {}\n            xmax = {}\n            text = "{}"\n'
POINT = '        points [{}]:\n            number = {}\n            mark = "{}"\n'


class Tier(NamedTuple):
    kind: str
    """`IntervalTier` or `TextTier`."""
    columns: dict[str, list]
    """Columns `text`, `start`, and `end` of the intervals, or `text` and `pos` of the points, as in `SCHEMAS`."""


class TextGrid(NamedTuple):
    xmin: float
    xmax: float
    tiers: dict[str, Tier]


def _decode(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"ooBinaryFile"):
        raise ValueError(f"Binary TextGrids are not supported, save {path} as a text file.")
    try:
        return raw.decode("utf-16" if raw.startswith((b"\xfe\xff", b"\xff\xfe")) else "utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is neither encoded in UTF-8 nor in UTF-16.") from error


def _numbers(tokens: list[str]) -> list[float]:
    values = list(map(float, tokens))  # Strings and flags raise a ValueError
    if not all(map(math.isfinite, values)):
        raise ValueError("times must be finite")
    return values


def _strings(tokens: list[str]) -> list[str]:
    if '"' in tokens or not all(map(str.startswith, tokens, itertools.repeat('"'))):
        raise ValueError(f"expected a string, got {next(t for t in tokens if t == '"' or t[0] != '"')!r}")
    return [token[1:-1].replace('""', '"') for token in tokens]


def _count(token: str) -> int:
    if (count := int(token)) < 0:
        raise ValueError(f"negative size {count}")
    return count


def _parse(tokens: list[str]) -> TextGrid:
    """Parse the tokens of a TextGrid, which must hold exactly as many tiers and entries as it declares."""
    if tokens[:2] != ['"ooTextFile"', '"TextGrid"']:
        raise ValueError("not a TextGrid in the long or short text format")
    position = 2

    def take(n: int) -> list[str]:
        nonlocal position
        if position + n > len(tokens):
            raise ValueError("unexpected end of file")
        position += n
        return tokens[position - n : position]

    xmin, xmax = _numbers(take(2))
    match take(1):
        case ["<exists>"]:
            num_tiers = _count(*take(1))
        case ["<absent>"]:
            num_tiers = 0
        case flag:
            raise ValueError(f"expected <exists> or <absent>, got {flag[0]!r}")
    tiers = {}
    for _ in range(num_tiers):
        kind, name = _strings(take(2))
        _numbers(take(2))  # The tier xmin and xmax, which are those of the TextGrid when written by Praat
        size = _count(*take(1))
        if kind == INTERVAL_TIER:
            block = take(3 * size)
            columns = {"text": _strings(block[2::3]), "start": _numbers(block[0::3]), "end": _numbers(block[1::3])}
        elif kind in {POINT_TIER, "PointTier"}:  # PointTier is written by praat-textgrids
            block, kind = take(2 * size), POINT_TIER
            columns = {"text": _strings(block[1::2]), "pos": _numbers(block[0::2])}
        else:
            raise ValueError(f"unsupported tier class {kind!r}")
        if name in tiers:
            raise ValueError(f"duplicate tier name {name!r}")
        tiers[name] = Tier(kind, columns)
    if position != len(tokens):
        raise ValueError("more entries than declared")
    return TextGrid(xmin, xmax, tiers)


def _read_textgrids(paths: list[Path]) -> list[TextGrid]:
    """Read TextGrids in the long or short text format of Praat, encoded in UTF-8 or UTF-16."""
    token = pl.element()
    skipped = token.str.starts_with("!") | (token.str.starts_with("[") & token.str.ends_with("]"))
    tokens = (
        pl.Series([_decode(p) for p in paths], dtype=pl.String)
        .str.extract_all(TOKEN)
        .list.eval(token.filter(~skipped))
        .to_list()
    )
    grids = []
    for path, file_tokens in zip(paths, tokens, strict=True):
        try:
            grids.append(_parse(file_tokens))
        except ValueError as error:
            raise ValueError(f"Invalid TextGrid {path}: {error}.") from None
    return grids


def read_textgrid(path: str | Path) -> dict[str, pl.DataFrame]:
    """Read a TextGrid file, or every TextGrid file in a directory and its subdirectories.

    The TextGrids must be in the long or short text format of Praat, encoded in UTF-8 or UTF-16. The file ids are
    the stems of the files, so they must be unique.

    Arguments:
        path: Path to a TextGrid file or to a directory

    Returns:
        Mapping between tier names and DataFrames sorted by file id. Interval tiers have columns `text`, `start`,
            `end`, and `fileid`, and point tiers `text`, `pos`, and `fileid`. Empty labels are replaced by `SIL`.

    """
    path = Path(path)
    if path.is_file():
        paths = [path]
    elif path.is_dir():
        paths = sorted(path.rglob("*.TextGrid"), key=lambda p: (p.stem, p))
        if not paths:
            raise ValueError(f"No TextGrid files found in directory {path}")
        if duplicates := sorted({a.stem for a, b in itertools.pairwise(paths) if a.stem == b.stem}):
            raise ValueError(f"Duplicate TextGrid file ids in {path}: {len(duplicates)}, such as {duplicates[:5]}.")
    else:
        raise ValueError(f"Path is neither a TextGrid file nor a directory: {path}")
    tiers: dict[str, Tier] = {}
    for p, grid in zip(paths, _read_textgrids(paths), strict=True):
        for name, tier in grid.tiers.items():
            merged = tiers.setdefault(name, Tier(tier.kind, {column: [] for column in [*tier.columns, "fileid"]}))
            if merged.kind != tier.kind:
                raise ValueError(f"Tier {name!r} is an interval tier in some TextGrids and a point tier in others.")
            for column, values in tier.columns.items():
                merged.columns[column].extend(values)
            merged.columns["fileid"].extend(itertools.repeat(p.stem, len(tier.columns["text"])))
    return {
        name: pl.DataFrame(columns, schema=SCHEMAS[kind] | {"fileid": pl.String}).with_columns(
            pl.col("text").replace("", "SIL")
        )
        for name, (kind, columns) in tiers.items()
    }


def _fill_gaps(tier: Tier, xmin: float, xmax: float) -> list[tuple[float, float, str]]:
    """Intervals `(start, end, text)` of a tier, with gaps filled by empty intervals, from `xmin` to `xmax`."""
    filled, previous = [], xmin
    for start, end, text in zip(tier.columns["start"], tier.columns["end"], tier.columns["text"], strict=True):
        if not (previous <= start < end and math.isfinite(end)):
            raise ValueError(
                f"invalid interval ({start}, {end}): intervals must last more than 0 s, must not overlap, and must "
                f"not start before the TextGrid (at {xmin} s)"
            )
        if start > previous:
            filled.append((previous, start, ""))
        filled.append((start, end, text))
        previous = end
    if previous < xmax:
        filled.append((previous, xmax, ""))
    return filled


def _format_entries(tier: Tier, xmin: float, xmax: float) -> str:
    """Format the intervals or points of a tier. The gaps between the sorted intervals are filled."""
    if tier.kind == INTERVAL_TIER:
        intervals = _fill_gaps(tier, xmin, xmax)
        return f"        intervals: size = {len(intervals)}\n" + "".join(
            INTERVAL.format(i, start, end, text.replace('"', '""'))
            for i, (start, end, text) in enumerate(intervals, start=1)
        )
    if min(tier.columns["pos"], default=xmin) < xmin:
        raise ValueError(f"points must not be before the TextGrid (at {xmin} s)")
    return f"        points: size = {len(tier.columns['pos'])}\n" + "".join(
        POINT.format(i, pos, text.replace('"', '""'))
        for i, (pos, text) in enumerate(zip(tier.columns["pos"], tier.columns["text"], strict=True), start=1)
    )


def _format(grid: TextGrid) -> str:
    """Format a TextGrid in the long text format, with every tier spanning the whole TextGrid."""
    xmin, xmax, tiers = grid
    out = [HEADER.format(xmin, xmax, len(tiers)), "item []:\n"]
    for index, (name, tier) in enumerate(tiers.items(), start=1):
        try:
            entries = _format_entries(tier, xmin, xmax)
        except ValueError as error:
            raise ValueError(f"tier {name!r}: {error}") from None
        out.extend((TIER.format(index, tier.kind, name.replace('"', '""'), xmin, xmax), entries))
    return "".join(out)


def _add_interval_tiers(outdir: Path, new_tiers: dict[str, dict[str, dict[str, list]]]) -> None:
    """Add interval tiers to the TextGrids in `outdir`, each file being read and written once.

    `new_tiers` maps tier names to file ids to sorted intervals (columns `text`, `start`, and `end`). The other tiers
    of existing files are kept, so these files must be readable by `read_textgrid`, and a tier with the same name is
    replaced. Each file starts at 0 (or before if it already did), and ends at its previous end or at the end of its
    last interval or point, whichever is later. Nothing is written if any interval is invalid, and each file is
    replaced atomically.
    """
    outdir.mkdir(parents=True, exist_ok=True)
    paths = {fileid: outdir / f"{fileid}.TextGrid" for intervals in new_tiers.values() for fileid in intervals}
    existing = [path for path in paths.values() if path.is_file()]
    grids = dict(zip(existing, _read_textgrids(existing), strict=True))
    contents = {}
    for fileid, path in paths.items():
        xmin, xmax, tiers = grids.get(path, TextGrid(0.0, 0.0, {}))
        xmin = min(xmin, 0.0)
        tiers |= {  # A replaced tier keeps its position
            name: Tier(INTERVAL_TIER, columns[fileid]) for name, columns in new_tiers.items() if fileid in columns
        }
        ends = (tier.columns["end" if tier.kind == INTERVAL_TIER else "pos"] for tier in tiers.values())
        xmax = max(xmax, *(max(times, default=xmax) for times in ends))
        try:
            contents[path] = _format(TextGrid(xmin, xmax, tiers))
        except ValueError as error:
            raise ValueError(f"Cannot write {path}, {error}.") from None
    for path, content in contents.items():
        tmp = path.with_name(f"{path.name}.tmp")  # Do not lose existing tiers if interrupted while writing
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)


def _runs(seqs: Phones | Units, *, step_in_ms: int) -> dict[str, dict[str, list]]:
    """Intervals (columns `text`, `start`, and `end`) of the runs of identical tokens of each sequence."""
    if empty := sorted(file for file, sequence in seqs.items() if not sequence):
        raise ValueError(f"Cannot write empty sequences to TextGrid: {len(empty)} files, such as {empty[:5]}.")
    intervals = {}
    for fileid, sequence in seqs.items():
        runs = [(text, sum(1 for _ in run)) for text, run in itertools.groupby(map(str, sequence))]
        bounds = [n * step_in_ms / 1000 for n in itertools.accumulate((length for _, length in runs), initial=0)]
        intervals[fileid] = {"text": [text for text, _ in runs], "start": bounds[:-1], "end": bounds[1:]}
    return intervals


def write_textgrids(seqs: Phones | Units, /, outdir: str | Path, *, tier_name: str, step_in_ms: int) -> None:
    """Write the given sequences of tokens as TextGrid files in `outdir`, one per file and in the tier `tier_name`.

    Consecutive identical tokens are merged into a single interval. If a TextGrid file already exists, the tier is
    added to it, or replaces the tier with the same name.
    """
    _add_interval_tiers(Path(outdir), {tier_name: _runs(seqs, step_in_ms=step_in_ms)})


def df_to_textgrids(
    df: pl.DataFrame,
    outdir: str | Path,
    *,
    file_col: str,
    begin_col: str,
    end_col: str,
    label_col: str,
    tier_name: str,
) -> None:
    """Write the intervals of a DataFrame as TextGrid files in `outdir`, one per file and in the tier `tier_name`.

    Times are rounded to the microsecond. The intervals of a file must not overlap, and the gaps between them are
    filled with empty intervals. If a TextGrid file already exists, the tier is added to it, or replaces the tier
    with the same name.
    """
    intervals = df.select(
        fileid=pl.col(file_col).cast(pl.String),
        text=pl.col(label_col).cast(pl.String),
        start=pl.col(begin_col).cast(pl.Float64).round(6),
        end=pl.col(end_col).cast(pl.Float64).round(6),
    )
    if intervals.null_count().sum_horizontal().item():
        raise ValueError("Cannot write TextGrid intervals with null file ids, labels, or times.")
    files = intervals.sort("fileid", "start").partition_by("fileid", as_dict=True, include_key=False)
    _add_interval_tiers(
        Path(outdir), {tier_name: {fileid: group.to_dict(as_series=False) for (fileid,), group in files.items()}}
    )


def rttm_to_textgrids(source: str | Path, outdir: str | Path, *, tier_name: str) -> None:
    """Write the speaker turns of an RTTM file as TextGrid files in `outdir`, in the tier `tier_name`."""
    df_to_textgrids(
        read_rttm(source).with_columns((pl.col("Turn Onset") + pl.col("Turn Duration")).alias("Turn Offset")),
        outdir,
        file_col="File ID",
        begin_col="Turn Onset",
        end_col="Turn Offset",
        label_col="Speaker Name",
        tier_name=tier_name,
    )


def cli(argv: list[str] | None = None) -> None:
    """Command-line entry point for exporting gold phones, units, and predicted phones to TextGrid."""
    parser = argparse.ArgumentParser(
        prog="discophon.textgrid",
        description="Export the gold phones, the units, and the phones predicted by the many-to-one mapping to "
        "TextGrids, in {outdir}/{language}/{split}/{file}.TextGrid",
    )
    parser.add_argument("dataset", type=Path, help="Path to the benchmark dataset")
    parser.add_argument("predictions", type=Path, help="Path to the directory with the discrete units")
    parser.add_argument("outdir", type=Path, help="Output directory")
    args = parser.parse_args(argv)

    for language, split in tqdm(available_languages_and_splits_for_units(args.predictions)):
        if split not in EVALUATED_SPLITS:
            continue
        units = read_submitted_units(args.predictions / units_filename(language, split))
        phones = read_gold_annotations(args.dataset / "alignment" / alignment_filename(language, split))
        cooccurrence = cooccurrence_matrix(units, phones, n_units=DEFAULT_N_UNITS, n_phonemes=language.n_phonemes)
        predictions = phone_assignments(units, cooccurrence, kind="many-to-one")
        _add_interval_tiers(
            args.outdir / language.iso_639_3 / split,
            {
                "phones": _runs(phones, step_in_ms=STEP_PHONES),
                "units": _runs(units, step_in_ms=STEP_UNITS),
                "predictions": _runs(predictions, step_in_ms=STEP_UNITS),
            },
        )


if __name__ == "__main__":
    cli()
