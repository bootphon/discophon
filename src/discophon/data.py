"""Data loading and writing utilities."""

import itertools
import re
from pathlib import Path
from typing import Literal

import polars as pl

from discophon.languages import Language, all_languages

__all__ = [
    "DEFAULT_N_UNITS",
    "STEP_PHONES",
    "STEP_UNITS",
    "Phones",
    "Splits",
    "Units",
    "alignment_filename",
    "df_to_textgrids",
    "item_filename",
    "manifest_filename",
    "read_gold_annotations",
    "read_rttm",
    "read_scores",
    "read_submitted_units",
    "read_textgrid",
    "rttm_to_textgrids",
    "units_filename",
    "write_textgrids",
]

type Splits = Literal["all", "train-10min", "train-1h", "train-10h", "dev", "test"]
"""Type of the dataset splits. `all` holds every audio file of a language, the other splits link to some of them."""

type Units = dict[str, list[int]]
"""Type of the discrete units: dictionary mapping file identifiers to lists of integers."""

type Phones = dict[str, list[str]]
"""Type of the gold or predicted phones: dictionary mapping file identifiers to list of strings."""

STEP_PHONES = 10
"""Constant step in ms between consecutive phone annotations. Override it in function parameters only if you
use new annotations built differently."""

STEP_UNITS = 20
"""Default step in ms between consecutive units. Corresponds to 50 Hz model. Can be overridden easily."""

DEFAULT_N_UNITS = 256
"""Default number of distinct units in the many-to-one evaluation."""

SAMPLE_RATE = 16_000
FILE, ONSET, OFFSET, PHONE, UNITS = "#file", "onset", "offset", "#phone", "units"

INTERVAL_TIER, POINT_TIER = "IntervalTier", "TextTier"
TEXTGRID_KEY_VALUE = r'(?m)^[ \t]*(?:intervals: |points: )?\w+ = (?:"(?:[^"]|"")*"[ \t\r]*$|[^"\s]\S*)'
"""A `key = value` line of a TextGrid in the long text format, whose value is a number or a quoted string. Strings
can span several lines and hold quotes escaped by doubling them, and must end their line: a string that is not
closed is skipped, and the TextGrid is then malformed. Interval and point counts have keys `intervals: size` and
`points: size`."""
TEXTGRID_HEADER_PATTERN = (
    r'File type = "ooTextFile"\s+Object class = "TextGrid"\s+xmin = \S+\s+xmax = \S+\s+'
    r"tiers\? (?:<absent>|<exists>\s+size = \d+)"
)
TEXTGRID_HEADER = (
    'File type = "ooTextFile"\nObject class = "TextGrid"\n\n'
    "xmin = 0.0\nxmax = {}\ntiers? <exists>\nsize = {}\nitem []:\n"
)
TEXTGRID_TIER = (
    '    item [{}]:\n        class = "{}"\n        name = "{}"\n'
    "        xmin = 0.0\n        xmax = {}\n        {}: size = {}\n"
)
TEXTGRID_INTERVAL = '        intervals [{}]:\n            xmin = {}\n            xmax = {}\n            text = "{}"\n'
TEXTGRID_POINT = '        points [{}]:\n            number = {}\n            mark = "{}"\n'


def units_filename(language: Language, split: str) -> str:
    """Filename for the predicted units of a (language, split) pair."""
    return f"units-{language.iso_639_3}-{split}.jsonl"


def alignment_filename(language: Language, split: str) -> str:
    """Filename for the gold phone alignment of a (language, split) pair."""
    return f"alignment-{language.iso_639_3}-{split}.txt"


def item_filename(language: Language, split: str, *, kind: str) -> str:
    """Filename for the ABX item file of a (language, split, kind) triple."""
    return f"{kind}-{language.iso_639_3}-{split}.item"


def manifest_filename(language: Language, split: str) -> str:
    """Filename for the audio manifest of a (language, split) pair."""
    return f"manifest-{language.iso_639_3}-{split}.csv"


def read_rttm(source: str | Path) -> pl.DataFrame:
    """Read an RTTM file, with one column per field of the format."""
    return pl.read_csv(
        source,
        has_header=False,
        new_columns=[
            "Type",
            "File ID",
            "Channel ID",
            "Turn Onset",
            "Turn Duration",
            "Orthography Field",
            "Speaker Type",
            "Speaker Name",
            "Confidence Score",
            "Signal Lookahead Time",
        ],
        separator=" ",
        schema_overrides={
            "Type": pl.String,
            "File ID": pl.String,
            "Turn Onset": pl.Float64,
            "Turn Duration": pl.Float64,
            "Speaker Name": pl.String,
        },
        null_values="<NA>",
    )


def _unquote(expr: pl.Expr) -> pl.Expr:
    """Remove the quotes around a string written in a TextGrid, and unescape the quotes inside it."""
    return expr.str.strip_prefix('"').str.strip_suffix('"').str.replace_all('""', '"', literal=True)


def _decode_textgrid(path: Path) -> str:
    raw = path.read_bytes()
    if raw.startswith(b"ooBinaryFile"):
        raise ValueError(f"Binary TextGrids are not supported, save {path} as a text file.")
    try:
        content = raw.decode("utf-16" if raw.startswith((b"\xfe\xff", b"\xff\xfe")) else "utf-8-sig")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path} is neither encoded in UTF-8 nor in UTF-16.") from error
    if not re.match(TEXTGRID_HEADER_PATTERN, content):
        raise ValueError(f"{path} is not a TextGrid in the long text format.")
    return content


def _read_textgrids(paths: list[Path]) -> dict[str, pl.DataFrame]:
    """Read TextGrids in the long text format, with the stem of each path as file id and the labels as written.

    Interval tiers have columns `fileid`, `text`, `start`, and `end`, and point tiers `fileid`, `text`, and `pos`.
    Each file must hold as many tiers, intervals, and points as it declares.
    """
    key, value = pl.col("key"), pl.col("value")
    tokens = (
        pl.DataFrame(
            {"fileid": [p.stem for p in paths], "content": [_decode_textgrid(p) for p in paths]},
            schema={"fileid": pl.String, "content": pl.String},
        )
        .select("fileid", token=pl.col("content").str.extract_all(TEXTGRID_KEY_VALUE))
        .explode("token")
        .select("fileid", pl.col("token").str.strip_chars().str.splitn(" = ", 2))
        .unnest("token")
        .rename({"field_0": "key", "field_1": "value"})
        .with_columns(  # The class and the name of a tier come before its intervals or points
            kind=pl.when(key == "class").then(_unquote(value)).forward_fill(),
            tier=pl.when(key == "name").then(_unquote(value)).forward_fill(),
        )
    )
    intervals = (
        tokens.with_columns(
            start=pl.when(key.shift(2) == "xmin").then(value.shift(2)).cast(pl.Float64, strict=False),
            end=pl.when(key.shift(1) == "xmax").then(value.shift(1)).cast(pl.Float64, strict=False),
        )
        .filter(key == "text", pl.col("kind") == INTERVAL_TIER)
        .select("fileid", "tier", text=_unquote(value), start="start", end="end")
    )
    points = (
        tokens.with_columns(pos=pl.when(key.shift(1) == "number").then(value.shift(1)).cast(pl.Float64, strict=False))
        .filter(key == "mark", pl.col("kind") == POINT_TIER)
        .select("fileid", "tier", text=_unquote(value), pos="pos")
    )
    names = tokens.filter(key == "name").select("fileid", "tier", "kind")
    if not (duplicates := names.filter(names.select("fileid", "tier").is_duplicated())).is_empty():
        raise ValueError(f"Duplicate tier names in TextGrids: {duplicates.unique(maintain_order=True).rows()[:5]}.")
    kinds = names.unique(["tier", "kind"], maintain_order=True).select("tier", "kind")
    if not (unsupported := kinds.filter(~pl.col("kind").is_in([INTERVAL_TIER, POINT_TIER]))).is_empty():
        raise ValueError(f"Unsupported TextGrid tiers: {unsupported.rows()[:5]}.")
    if conflicting := kinds.filter(pl.col("tier").is_duplicated())["tier"].unique(maintain_order=True).to_list():
        raise ValueError(f"Tiers that are interval tiers in some TextGrids and point tiers in others: {conflicting}.")
    num_entries = pl.concat([intervals.select("fileid", "tier"), points.select("fileid", "tier")]).group_by(
        "fileid", "tier"
    )
    malformed = pl.concat(
        [
            intervals.filter(pl.any_horizontal(pl.col("start", "end").is_null()))["fileid"],
            points.filter(pl.col("pos").is_null())["fileid"],
            names.join(
                tokens.filter(key.is_in(["intervals: size", "points: size"])).select("fileid", "tier", "value"),
                on=["fileid", "tier"],
                how="left",
            )
            .join(num_entries.len(), on=["fileid", "tier"], how="left")
            .filter(value.cast(pl.Int64, strict=False).ne_missing(pl.col("len").fill_null(0)))["fileid"],
            tokens.filter(key == "size")
            .join(names.group_by("fileid").len(), on="fileid", how="left")
            .filter(value.cast(pl.Int64, strict=False).ne_missing(pl.col("len").fill_null(0)))["fileid"],
        ]
    ).unique(maintain_order=True)
    if not malformed.is_empty():
        raise ValueError(f"Malformed TextGrids: {len(malformed)} files, such as {malformed.head(5).to_list()}.")
    return {
        tier: (intervals if kind == INTERVAL_TIER else points).filter(pl.col("tier") == tier).drop("tier")
        for tier, kind in kinds.iter_rows()
    }


def read_textgrid(path: str | Path) -> dict[str, pl.DataFrame]:
    """Read a TextGrid file, or every TextGrid file in a directory and its subdirectories.

    The TextGrids must be in the long text format of Praat, encoded in UTF-8 or UTF-16. The file ids are the
    stems of the files, so they must be unique.

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
    return {
        tier: df.select(pl.col("text").replace("", "SIL"), pl.exclude("text", "fileid"), "fileid")
        for tier, df in _read_textgrids(paths).items()
    }


def _add_textgrid_tier(intervals: pl.DataFrame, outdir: Path, *, tier_name: str) -> None:
    """Add intervals (`fileid`, `text`, `start`, `end`) as the tier `tier_name` of the TextGrids in `outdir`.

    The other tiers of existing files are kept, so these files must be readable by `read_textgrid`, and a tier with
    the same name is replaced. Times are rounded to
    the microsecond. The gaps between intervals are filled with empty intervals, so that each interval tier spans
    the whole TextGrid, from 0 to the end of its last interval or point.
    """
    if intervals.null_count().sum_horizontal().item():
        raise ValueError("Cannot write TextGrid intervals with null file ids, labels, or times.")
    outdir.mkdir(parents=True, exist_ok=True)
    existing = [
        p for f in intervals["fileid"].unique(maintain_order=True) if (p := outdir / f"{f}.TextGrid").is_file()
    ]
    tiers = _read_textgrids(existing) | {tier_name: intervals}
    times = ["start", "end", "pos"]
    entries = pl.concat(
        [
            df.select(
                "fileid",
                "text",
                *[
                    pl.col(t).cast(pl.Float64).round(6) if t in df.columns else pl.lit(None, pl.Float64).alias(t)
                    for t in times
                ],
                tier=pl.lit(name),
                rank=pl.lit(rank),
            )
            for rank, (name, df) in enumerate(tiers.items())
        ]
    ).with_columns(xmax=pl.max_horizontal("end", "pos").max().over("fileid"))
    is_interval = pl.col("pos").is_null()
    interval_entries = (
        entries.filter(is_interval)
        .sort("fileid", "rank", "start")
        .with_columns(prev=pl.col("end").shift(1, fill_value=0.0).over("fileid", "rank"))
    )
    invalid = interval_entries.filter((pl.col("start") < pl.col("prev")) | (pl.col("end") <= pl.col("start"))).select(
        "fileid", "tier", "start", "end"
    )
    if not invalid.is_empty():
        raise ValueError(
            "TextGrid intervals must not start before 0, must last more than 0 s, and must not overlap, but got "
            f"{invalid.height} invalid intervals, such as {invalid.rows()[:5]}."
        )
    gaps = pl.concat(
        [
            interval_entries.filter(pl.col("start") > pl.col("prev")).with_columns(
                text=pl.lit(""), start="prev", end="start"
            ),
            interval_entries.filter(
                pl.col("end") == pl.col("end").max().over("fileid", "rank"), pl.col("end") < pl.col("xmax")
            ).with_columns(text=pl.lit(""), start="end", end="xmax"),
        ]
    )
    tier_header = pl.format(
        TEXTGRID_TIER,
        pl.int_range(1, pl.len() + 1).over("fileid"),
        pl.when("is_interval").then(pl.lit(INTERVAL_TIER)).otherwise(pl.lit(POINT_TIER)),
        pl.col("tier").str.replace_all('"', '""', literal=True),
        "xmax",
        pl.when("is_interval").then(pl.lit("intervals")).otherwise(pl.lit("points")),
        "len",
    )
    position, text = (
        pl.int_range(1, pl.len() + 1).over("fileid", "rank"),
        pl.col("text").str.replace_all('"', '""', literal=True),
    )
    files = (
        pl.concat([interval_entries.drop("prev"), gaps.drop("prev"), entries.filter(~is_interval)])
        .sort("fileid", "rank", pl.coalesce("start", "pos"))
        .with_columns(
            body=pl.when(is_interval)
            .then(pl.format(TEXTGRID_INTERVAL, position, "start", "end", text))
            .otherwise(pl.format(TEXTGRID_POINT, position, "pos", text))
        )
        .group_by("fileid", "rank", maintain_order=True)
        .agg(pl.first("tier", "xmax"), is_interval.first().alias("is_interval"), pl.len(), pl.col("body").str.join())
        .with_columns(body=tier_header + pl.col("body"))
        .group_by("fileid", maintain_order=True)
        .agg(pl.first("xmax"), pl.len(), pl.col("body").str.join())
        .select("fileid", content=pl.format(TEXTGRID_HEADER, "xmax", "len") + pl.col("body"))
    )
    for fileid, content in files.iter_rows():
        path = outdir / f"{fileid}.TextGrid"
        tmp = path.with_name(f"{path.name}.tmp")  # Do not lose existing tiers if interrupted while writing
        tmp.write_text(content, encoding="utf-8")
        tmp.replace(path)


def write_textgrids(seqs: Phones | Units, /, outdir: str | Path, *, tier_name: str, step_in_ms: int) -> None:
    """Write the given sequences of tokens as TextGrid files in `outdir`, one per file and in the tier `tier_name`.

    Consecutive identical tokens are merged into a single interval. If a TextGrid file already exists, the tier is
    added to it, or replaces the tier with the same name.
    """
    if empty := sorted(file for file, sequence in seqs.items() if not sequence):
        raise ValueError(f"Cannot write empty sequences to TextGrid: {len(empty)} files, such as {empty[:5]}.")
    intervals = (
        pl.DataFrame(
            {"fileid": list(seqs), "text": [list(map(str, sequence)) for sequence in seqs.values()]},
            schema={"fileid": pl.String, "text": pl.List(pl.String)},
        )
        .explode("text")
        .with_columns(run=pl.col("text").rle_id().over("fileid"))
        .group_by("fileid", "run", maintain_order=True)
        .agg(pl.first("text"), pl.len())
        .with_columns(end=pl.col("len").cum_sum().over("fileid") * step_in_ms)
        .select("fileid", "text", start=(pl.col("end") - pl.col("len") * step_in_ms) / 1000, end=pl.col("end") / 1000)
    )
    _add_textgrid_tier(intervals, Path(outdir), tier_name=tier_name)


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

    The intervals of a file must not overlap, and the gaps between them are filled with empty intervals. If a
    TextGrid file already exists, the tier is added to it, or replaces the tier with the same name.
    """
    intervals = df.select(
        fileid=pl.col(file_col).cast(pl.String),
        text=pl.col(label_col).cast(pl.String),
        start=pl.col(begin_col).cast(pl.Float64),
        end=pl.col(end_col).cast(pl.Float64),
    )
    _add_textgrid_tier(intervals, Path(outdir), tier_name=tier_name)


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


def num_invalid_rows(df: pl.DataFrame, *, step_in_ms: int) -> int:
    """For each file, the first entry starts at 0 and each subsequent entry starts where the previous has ended."""
    incorrect_duration = ~(step_in_ms / 1000 <= pl.col(OFFSET) - pl.col(ONSET))
    return int(
        df.with_columns(pl.col(OFFSET).shift(1).over(FILE).alias(f"prev_{OFFSET}"))
        .with_columns(
            pl.when(pl.col(f"prev_{OFFSET}").is_null())
            .then((pl.col(ONSET) != 0) | incorrect_duration)
            .otherwise((pl.col(ONSET) != pl.col(f"prev_{OFFSET}")) | incorrect_duration)
            .alias("invalid")
        )["invalid"]
        .sum()
    )


def decimal_series_is_integer(series: pl.Series) -> bool:
    return (
        series.cast(pl.String)
        .str.split_exact(".", 1)
        .struct.rename_fields(["integer", "fractional"])
        .struct.field("fractional")
        .str.replace_all("0", "")
        .eq("")
        .all()
    )


def read_gold_annotations_as_dataframe(source: str | Path) -> pl.DataFrame:
    df = pl.read_csv(source, separator=" ", columns=[FILE, ONSET, OFFSET, PHONE], schema_overrides=[pl.String] * 4)
    if df.null_count().sum_horizontal().item():
        raise ValueError("Invalid annotations: file IDs, onsets, offsets, and phones must not be null.")
    return df.with_columns(
        df[ONSET].str.to_decimal(inference_length=len(df)),
        df[OFFSET].str.to_decimal(inference_length=len(df)),
    ).sort(FILE, ONSET)


class AnnotationsError(ValueError):
    def __init__(self, step_in_ms: int) -> None:
        super().__init__(
            "Invalid annotations: each entry should start where the previous one has ended, "
            f"and last at least {step_in_ms} ms."
        )


def read_gold_annotations(source: str | Path, *, step_in_ms: int = STEP_PHONES) -> Phones:
    """Read the gold annotations and return a mapping between file names to the list of phonemes.

    There will be one phone every `step_in_ms` ms.

    Arguments:
        source: Path to the annotations file
        step_in_ms: Step in ms between each phone.

    Returns:
        Mapping between file ids and phones

    """
    phones_per_seconds = 1000 // step_in_ms
    if step_in_ms * phones_per_seconds != 1000:
        raise ValueError(f"step_in_ms={step_in_ms} is not valid, it should be a divisor of 1000.")
    df = read_gold_annotations_as_dataframe(source)
    if num_invalid_rows(df, step_in_ms=step_in_ms) > 0:
        raise AnnotationsError(step_in_ms)
    df = df.with_columns(num=(pl.col(OFFSET) - pl.col(ONSET)) * phones_per_seconds)
    if not decimal_series_is_integer(df["num"]):
        raise ValueError(f"Each phone should last a multiple of {step_in_ms} ms, but found some that don't.")
    return {
        audio: row[PHONE]
        for audio, row in (
            df.with_columns(pl.col("num").cast(pl.Int64))
            .with_columns(pl.col(PHONE).repeat_by("num"))
            .group_by(FILE, maintain_order=True)
            .agg(pl.col(PHONE).explode())
            .rows_by_key(FILE, named=True, unique=True)
            .items()
        )
    }


def read_submitted_units(source: str | Path) -> Units:
    """Read the units from a JSONL file. Requires fields named `file` ([`str`][]) and `units` (`list[int]`).

    Other fields are ignored.

    Arguments:
        source: Path to the units file

    Returns:
        Mapping between file ids and units

    Raises:
        ValueError: If a file appears more than once, or if some units are missing.

    """
    df = pl.read_ndjson(source, schema_overrides={"file": pl.String, UNITS: pl.List(pl.Int32)}).rename({"file": FILE})
    if duplicates := sorted(set(df.filter(pl.col(FILE).is_duplicated())[FILE])):
        raise ValueError(f"Duplicate files in {source}: {len(duplicates)} files, such as {duplicates[:5]}.")
    is_null = pl.col(UNITS).is_null() | pl.col(UNITS).list.eval(pl.element().is_null()).list.any()
    if missing := sorted(df.filter(is_null)[FILE]):
        raise ValueError(f"Missing or null units in {source}: {len(missing)} files, such as {missing[:5]}.")
    return {audio: row[UNITS] for audio, row in df.rows_by_key(FILE, named=True, unique=True).items()}


def read_scores(root: str | Path) -> pl.DataFrame:
    """Read the scores of the [artifacts dataset](https://huggingface.co/datasets/coml/discophon-artifacts).

    Reads every `{model}/{condition}/{folder}/{layer}/scores.jsonl` in the dataset, or in one model directory if
    `root` holds an `info.json`. The condition is `zero-shot` or `ft-{language}-{duration}`, and the folder is
    `many_to_one`, `one_to_one`, `continuous`, or `many_to_one-k{N}`.

    ABX scores within and across speakers are also averaged, under the name of the metric without the speaker
    condition. For example, `triphone_abx_continuous` averages `triphone_abx_continuous_within_speaker` and
    `triphone_abx_continuous_across_speaker`, and `phoneme_abx_discrete_any_context` averages
    `phoneme_abx_discrete_within_speaker_any_context` and `phoneme_abx_discrete_across_speaker_any_context`.

    Arguments:
        root: Path to the artifacts dataset, or to one model directory in it

    Returns:
        DataFrame with one row per score and columns `model`, `folder`, `ft_lang` (null for zero-shot),
            `duration` (`"0"` for zero-shot), `layer`, `split`, `language`, `language_split` (whether the
            language is a dev or test language), `metric`, and `score`, as stored (not in %).

    """
    root = Path(root).resolve()  # the model name is read from the path, so `.` must be resolved
    model_dirs = [root] if (root / "info.json").exists() else sorted(p.parent for p in root.glob("*/info.json"))
    paths = [p for model_dir in model_dirs for p in sorted(model_dir.glob("*/*/*/scores.jsonl"))]
    if not paths:
        raise ValueError(f"No scores in {root}.")
    scores = (
        pl.concat(
            [
                pl.read_ndjson(p).with_columns(
                    model=pl.lit(p.parts[-5]),
                    condition=pl.lit(p.parts[-4]),
                    folder=pl.lit(p.parts[-3]),
                    layer=pl.lit(int(p.parts[-2]), dtype=pl.Int64),
                )
                for p in paths
            ]
        )
        .with_columns(
            ft_lang=pl.col("condition").str.extract(r"^ft-([a-z]{3})-"),
            duration=pl.col("condition").str.extract(r"^ft-[a-z]{3}-(.+)$").fill_null("0"),
            language_split=pl.col("language").replace_strict({lang.iso_639_3: lang.split for lang in all_languages()}),
        )
        .select(
            "model", "folder", "ft_lang", "duration", "layer", "split", "language", "language_split", "metric", "score"
        )
    )
    keys = [c for c in scores.columns if c != "score"]
    if not (duplicates := scores.filter(scores.select(keys).is_duplicated())).is_empty():
        raise ValueError(f"Duplicate scores in {root}: {duplicates.row(0, named=True)}.")
    speakers = r"^(.+_abx_(?:discrete|continuous))_(?:within|across)_speaker(.*)$"
    averaged = (
        scores.filter(pl.col("metric").str.contains(speakers))
        .with_columns(pl.col("metric").str.replace(speakers, "${1}${2}"))
        .group_by(keys, maintain_order=True)
        .agg(pl.mean("score"), n=pl.len())
    )
    if not averaged.filter(pl.col("n") != 2).is_empty():
        raise ValueError(f"ABX within speaker without across speaker (or the opposite) in {root}.")
    return pl.concat([scores, averaged.drop("n")]).sort(keys)
