import argparse
import json
import tomllib
from pathlib import Path
from typing import Any, NotRequired, TypedDict, get_type_hints

import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

from discophon.languages import all_languages
from discophon.paper import best_on_this_metric, read_scores


class ModelEntry(TypedDict):
    label: str
    url: str
    allow_partial: NotRequired[bool]


class SubmissionEntry(TypedDict):
    label: str
    track: str
    step_units: int
    url: str
    authors: str
    year: int
    description: str


REGISTRY_SCHEMAS = {"baseline": ModelEntry, "topline": ModelEntry, "submission": SubmissionEntry}
REGISTRY_FILES = {"baseline": "baselines.toml", "topline": "toplines.toml", "submission": "submissions.toml"}
SUBMISSION_TRACKS = {"many-to-one", "one-to-one"}
CATEGORIES = [{"key": "baseline", "label": "Baselines"}, {"key": "submission", "label": "Submissions"}]
METRICS = {
    "per": {"label": "PER", "arrow": "down"},
    "r_val": {"label": "R-value", "arrow": "up"},
    "f1": {"label": "F1", "arrow": "up"},
    "pnmi": {"label": "PNMI", "arrow": "up"},
    "triphone_abx_continuous": {"label": "ABX c.", "arrow": "down"},
}
DURATIONS = [{"key": "0", "label": "Zero-shot"}, {"key": "10h", "label": "Finetuned on 10h"}]
AVG_LANGUAGE = "__avg__"
TRACKS = {
    "many_to_one": {"metrics": ["per", "r_val", "f1", "pnmi", "triphone_abx_continuous"], "select_best_layer": True},
    "one_to_one": {"metrics": ["per", "r_val", "f1", "pnmi"], "select_best_layer": False},
}


def _validate_submission(key: str, entry: dict[str, Any], source: str) -> None:
    """Submission-only checks beyond the field schema: enumerated track and positive ints."""
    if entry["track"] not in SUBMISSION_TRACKS:
        raise ValueError(
            f"{source}: [{key}] field 'track' must be one of {sorted(SUBMISSION_TRACKS)}, got {entry['track']!r}."
        )
    for field in ("step_units", "year"):
        if entry[field] <= 0:
            raise ValueError(f"{source}: [{key}] field {field!r} must be a positive integer.")


def _validate_entry(key: str, entry: dict[str, Any], category: str, source: str) -> None:
    """Validate one registry section against its TypedDict schema, raising on the first problem."""
    schema = REGISTRY_SCHEMAS[category]
    types = get_type_hints(schema)  # {field: python type}, with NotRequired stripped
    for field in schema.__required_keys__:
        if field not in entry:
            raise ValueError(f"{source}: [{key}] is missing required field {field!r}.")
    for field, value in entry.items():
        if field not in types:
            raise ValueError(f"{source}: [{key}] has unknown field {field!r}.")
        expected = types[field]
        # bool is a subclass of int, so an int field must reject True/False explicitly.
        if not isinstance(value, expected) or (expected is int and isinstance(value, bool)):
            raise ValueError(
                f"{source}: [{key}] field {field!r} must be {expected.__name__}, got {type(value).__name__}."
            )
    if category == "submission":
        _validate_submission(key, entry, source)


def _read_registry(registry_dir: Path, category: str) -> dict[str, dict[str, Any]]:
    path = registry_dir / REGISTRY_FILES[category]
    if not path.exists():
        return {}
    with path.open("rb") as fh:
        data = tomllib.load(fh)
    for key, entry in data.items():
        if not isinstance(entry, dict):
            raise ValueError(f"{path.name}: [{key}] must be a table.")
        _validate_entry(key, entry, category, path.name)
    return data


def load_models(registry_dir: Path, category: str) -> dict[str, dict[str, Any]]:
    """Load baselines.toml or toplines.toml, stamping each entry with its category."""
    return {key: {**entry, "category": category} for key, entry in _read_registry(registry_dir, category).items()}


def load_submissions(registry_dir: Path, track: str) -> dict[str, dict[str, Any]]:
    """Discover submissions.toml entries for one track. A submission's TOML 'track' uses
    hyphens (many-to-one) while TRACKS keys use underscores (many_to_one); map between them."""
    track_label = track.replace("_", "-")
    return {
        key: {**entry, "category": "submission"}
        for key, entry in _read_registry(registry_dir, "submission").items()
        if entry["track"] == track_label
    }


def load_track_models(registry_dir: Path, track: str) -> dict[str, dict[str, Any]]:
    """Models shown on a track table: hardcoded-free baselines plus discovered submissions."""
    return {**load_models(registry_dir, "baseline"), **load_submissions(registry_dir, track)}


def model_link(key: str, model: dict[str, Any]) -> dict[str, Any]:
    if model["category"] == "submission":
        return {"href": f"#model-{key}", "external": False}
    return {"href": model["url"], "external": True}


def markers(name: str) -> tuple[str, str]:
    return (f"<!-- discophon-leaderboard:{name}:start -->", f"<!-- discophon-leaderboard:{name}:end -->")


def _inject(page_path: Path, name: str, body: str) -> Path:
    """Replace the text between the named markers in the page with body."""
    marker_start, marker_end = markers(name)
    page = page_path.read_text(encoding="utf-8")
    start, end = page.find(marker_start), page.find(marker_end)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"{page_path} must contain '{marker_start}' ... '{marker_end}' markers")
    updated = page[: start + len(marker_start)] + f"\n{body}\n" + page[end:]
    page_path.write_text(updated, encoding="utf-8")
    return page_path


def _language_index() -> dict[str, dict[str, str]]:
    return {lang.iso_639_3: {"name": lang.name, "split": lang.split} for lang in all_languages()}


def _splits_payload(index: dict[str, dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    splits = {"dev": [], "test": []}
    for iso, info in sorted(index.items(), key=lambda kv: kv[1]["name"]):
        splits[info["split"]].append({"iso": iso, "name": info["name"]})
    return splits


def _expected_lang_counts() -> dict[str, int]:
    counts: dict[str, int] = {"dev": 0, "test": 0}
    for info in _language_index().values():
        counts[info["split"]] += 1
    return counts


def _base_frame(scores_dir: Path, track: dict[str, Any], models: dict[str, dict[str, Any]]) -> pl.DataFrame:
    df = read_scores(scores_dir).filter(
        pl.col("split") == "test",
        pl.col("metric").is_in(track["metrics"]),
        pl.col("model").is_in(list(models)),
        pl.col("duration").is_in([d["key"] for d in DURATIONS]),
    )
    if track["select_best_layer"]:
        filtered = df.filter(pl.col("best_layer"))
        # Models with no dev-language data (e.g. evaluated only on test-split languages)
        # have best_layer=False everywhere and drop out entirely. Fall back to their
        # minimum layer so they still appear in the table.
        before_keys = df.select("model", "duration").unique()
        after_keys = filtered.select("model", "duration").unique()
        missing = before_keys.join(after_keys, on=["model", "duration"], how="anti")
        if len(missing) > 0:
            fallback_layers = (
                df.join(missing, on=["model", "duration"])
                .group_by(["model", "duration"])
                .agg(pl.min("layer").alias("min_layer"))
            )
            fallback = (
                df.join(fallback_layers, on=["model", "duration"])
                .filter(pl.col("layer") == pl.col("min_layer"))
                .drop("min_layer")
            )
            filtered = pl.concat([filtered, fallback])
        df = filtered
    return df.select("model", "layer", "duration", "language", "test_split", "metric", "score")


def _rows_with_avg(base: pl.DataFrame) -> pl.DataFrame:
    columns = ["model", "layer", "duration", "test_split", "language", "metric", "score", "std"]
    per_lang = (
        base.group_by(["model", "layer", "duration", "test_split", "language", "metric"], maintain_order=True)
        .agg(pl.mean("score"))
        .with_columns(std=pl.lit(None, dtype=pl.Float64))
        .select(columns)
    )
    expected = _expected_lang_counts()
    complete = (
        base.group_by(["model", "layer", "duration", "test_split"])
        .agg(pl.col("language").n_unique().alias("n_langs"))
        .with_columns(expected_count=pl.col("test_split").replace_strict(expected))
        .filter(pl.col("n_langs") == pl.col("expected_count"))
        .select(["model", "layer", "duration", "test_split"])
    )
    avg = (
        base.join(complete, on=["model", "layer", "duration", "test_split"])
        .group_by(["model", "layer", "duration", "test_split", "metric"], maintain_order=True)
        .agg(pl.mean("score"), pl.std("score").alias("std"))
        .with_columns(language=pl.lit(AVG_LANGUAGE))
        .select(columns)
    )
    return pl.concat([per_lang, avg]).with_columns(
        top=best_on_this_metric().over(["metric", "duration", "test_split", "language"])
    )


def _row_payload(row: dict[str, Any], metrics: list[str]) -> dict[str, Any]:
    scores, top, stds = {}, {}, {}
    for metric in metrics:
        score = row.get(f"score_{metric}")
        top_flag = row.get(f"top_{metric}")
        scores[metric] = score
        top[metric] = bool(top_flag) if score is not None else False
        stds[metric] = row.get(f"std_{metric}")
    return {
        "model": row["model"],
        "layer": row["layer"],
        "duration": row["duration"],
        "split": row["test_split"],
        "language": row["language"],
        "scores": scores,
        "top": top,
        "stds": stds,
    }


def check_language_coverage(scores_dir: Path, models: dict[str, dict[str, Any]]) -> None:
    """Raise ValueError if any non-partial model is missing languages for any split."""
    expected = _expected_lang_counts()
    full_models = {k: v for k, v in models.items() if not v.get("allow_partial")}
    if not full_models:
        return
    df = read_scores(scores_dir).filter(pl.col("split") == "test", pl.col("model").is_in(list(full_models)))
    for key in full_models:
        for split_key, n_expected in expected.items():
            n_langs = df.filter((pl.col("model") == key) & (pl.col("test_split") == split_key))["language"].n_unique()
            if n_langs < n_expected:
                raise ValueError(
                    f"Model {key!r} covers only {n_langs}/{n_expected} {split_key} languages. "
                    f"Add allow_partial=true to its registry entry to permit partial coverage."
                )


def build_payload(scores_dir: Path, track: dict[str, Any], models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    metrics = track["metrics"]
    index = _language_index()
    wide = _rows_with_avg(_base_frame(scores_dir, track, models)).pivot(
        on="metric",
        index=["model", "layer", "duration", "test_split", "language"],
        values=["score", "top", "std"],
    )
    rows = [_row_payload(r, metrics) for r in wide.iter_rows(named=True)]
    return {
        "metrics": [{"key": k, **METRICS[k]} for k in metrics],
        "models": [
            {"key": k, "label": v["label"], "category": v["category"], **model_link(k, v)} for k, v in models.items()
        ],
        "categories": CATEGORIES,
        "durations": DURATIONS,
        "splits": _splits_payload(index),
        "avg_key": AVG_LANGUAGE,
        "rows": rows,
    }


def build_track(track: str, scores_dir: Path, page_path: Path, templates_dir: Path, registry_dir: Path) -> Path:
    models = load_track_models(registry_dir, track)
    check_language_coverage(scores_dir, models)
    payload = build_payload(scores_dir, TRACKS[track], models)
    env = Environment(loader=FileSystemLoader(templates_dir), autoescape=select_autoescape(["html", "j2"]))
    element_id = f"discophon-{track.replace('_', '-')}"
    snippet = env.get_template("leaderboard.snippet.html.j2").render(
        metrics=payload["metrics"],
        data_json=json.dumps(payload),
        default_split="test",
        element_id=element_id,
    )
    _inject(page_path, track, snippet)
    submissions = load_submissions(registry_dir, track)
    descriptions = env.get_template("leaderboard.submission.md.j2").render(
        submissions=[{"key": key, **entry} for key, entry in submissions.items()]
    )
    return _inject(page_path, f"{track}:descriptions", descriptions)


def build_topline_payload(scores_dir: Path, models: dict[str, dict[str, Any]]) -> dict[str, Any]:
    track_metrics: list[str] = TRACKS["many_to_one"]["metrics"]  # ty:ignore[invalid-assignment]
    index = _language_index()
    base = (
        read_scores(scores_dir)
        .filter(
            pl.col("split") == "test",
            pl.col("metric").is_in(track_metrics),
            pl.col("model").is_in(list(models)),
        )
        .select("model", "layer", "duration", "language", "test_split", "metric", "score")
    )
    wide = _rows_with_avg(base).pivot(
        on="metric",
        index=["model", "layer", "duration", "test_split", "language"],
        values=["score", "top", "std"],
    )
    rows = [_row_payload(r, track_metrics) for r in wide.iter_rows(named=True)]
    return {
        "metrics": [{"key": k, **METRICS[k]} for k in track_metrics],
        "models": [
            {"key": k, "label": v["label"], "category": "topline", "href": v["url"], "external": True}
            for k, v in models.items()
        ],
        "categories": [{"key": "topline", "label": "Toplines"}],
        "durations": [{"key": "0", "label": "Zero-shot"}],
        "splits": _splits_payload(index),
        "avg_key": AVG_LANGUAGE,
        "rows": rows,
    }


def build_topline(scores_dir: Path, page_path: Path, templates_dir: Path, registry_dir: Path) -> Path:
    payload = build_topline_payload(scores_dir, load_models(registry_dir, "topline"))
    env = Environment(loader=FileSystemLoader(templates_dir), autoescape=select_autoescape(["html", "j2"]))
    snippet = env.get_template("leaderboard.snippet.html.j2").render(
        metrics=payload["metrics"],
        data_json=json.dumps(payload),
        default_split="test",
        element_id="discophon-topline",
    )
    return _inject(page_path, "topline", snippet)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path, help="directory of *.jsonl score files")
    parser.add_argument("templates", type=Path, help="templates directory")
    parser.add_argument("page", type=Path, help="markdown page to inject the leaderboard into (between markers)")
    parser.add_argument("--track", choices=list(TRACKS), default="many_to_one")
    parser.add_argument("--topline", action="store_true", help="build the interactive topline table instead")
    parser.add_argument(
        "--registry",
        type=Path,
        default=Path(),
        help="directory holding baselines.toml / toplines.toml / submissions.toml (default: cwd)",
    )
    args = parser.parse_args()
    if args.topline:
        build_topline(args.scores, args.page, args.templates, args.registry)
    else:
        build_track(args.track, args.scores, args.page, args.templates, args.registry)


if __name__ == "__main__":
    main()
