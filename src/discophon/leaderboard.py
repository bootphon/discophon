import argparse
import json
from pathlib import Path
from typing import Any

import polars as pl
from jinja2 import Environment, FileSystemLoader, select_autoescape

from discophon.languages import all_languages
from discophon.paper import best_on_this_metric, read_scores

# Each model is tagged "baseline" or "submission" so the leaderboard groups them
# under the matching category header (see CATEGORIES) within each duration block.
# Baselines carry a "checkpoint" URL: their name links out to the public HuggingFace
# checkpoint. Submissions instead link to an in-page description section (see model_link).
# Set allow_partial=True on any model that is intentionally evaluated on a subset of
# languages; the coverage check in build_track will then skip it.
MODELS = {
    "spidr-mmsulab": {
        "label": "SpidR MMS-ulab",
        "category": "baseline",
        "checkpoint": "https://huggingface.co/coml/spidr-mmsulab",
    },
    "spidr-vp20": {
        "label": "SpidR VP-20",
        "category": "baseline",
        "checkpoint": "https://huggingface.co/coml/spidr-vp20",
    },
    "hubert-mmsulab-it2": {
        "label": "HuBERT MMS-ulab",
        "category": "baseline",
        "checkpoint": "https://huggingface.co/coml/hubert-base-mmsulab",
    },
    "hubert-vp20-it2": {
        "label": "HuBERT VP-20",
        "category": "baseline",
        "checkpoint": "https://huggingface.co/coml/hubert-base-vp20",
    },
    "hubert-jpn": {
        "label": "HuBERT base (ja)",
        "category": "baseline",
        "checkpoint": "https://huggingface.co/rinna/japanese-hubert-base",
        "allow_partial": True,
    },
}

CATEGORIES = [
    {"key": "baseline", "label": "Baselines"},
    {"key": "submission", "label": "Submissions"},
]

# Topline models do not follow the benchmark's full evaluation protocol.
# They link to their checkpoint or paper only (no in-page description).
TOPLINE_MODELS = {
    "mms": {
        "label": "MMS",
        "href": "https://huggingface.co/facebook/mms-300m",
    },
    "xeus": {
        "label": "XEUS",
        "href": "https://huggingface.co/microsoft/XEUS",
    },
    "hubert-base": {
        "label": "HuBERT base",
        "href": "https://huggingface.co/facebook/hubert-base-ls960",
    },
}


# Where a model's name links in the leaderboard. Baselines link out to their public
# HuggingFace checkpoint; submissions link to an in-page description section whose
# heading id is set with attr_list in leaderboard/index.md, e.g. `### ... { #model-<key> }`.
def model_link(key: str, model: dict[str, Any]) -> dict[str, Any]:
    if model["category"] == "baseline":
        return {"href": model["checkpoint"], "external": True}
    return {"href": f"#model-{key}", "external": False}


METRICS = {
    "per": {"label": "PER", "arrow": "down"},
    "r_val": {"label": "R-value", "arrow": "up"},
    "f1": {"label": "F1", "arrow": "up"},
    "pnmi": {"label": "PNMI", "arrow": "up"},
    "triphone_abx_continuous": {"label": "ABX c.", "arrow": "down"},
}

DURATIONS = [
    {"key": "0", "label": "Zero-shot"},
    {"key": "10h", "label": "Finetuned on 10h"},
]

AVG_LANGUAGE = "__avg__"

# Per-track configuration. "metrics" lists which METRICS columns the track shows: the
# one-to-one scores carry no ABX (it is a representation metric, identical across tracks),
# so that track drops the ABX column. "select_best_layer" re-picks the best layer per model
# from all layers (read_scores flags it via ABX on dev); the one-to-one scores are merged
# from already-best-layer runs, so that track keeps every row as is.
TRACKS = {
    "many_to_one": {
        "metrics": ["per", "r_val", "f1", "pnmi", "triphone_abx_continuous"],
        "select_best_layer": True,
    },
    "one_to_one": {
        "metrics": ["per", "r_val", "f1", "pnmi"],
        "select_best_layer": False,
    },
}


# The generated table for each track is injected between its own pair of markers in
# the target markdown page; everything outside them stays authored by hand. Markers are
# track-scoped so the many-to-one and one-to-one tables can share a single page.
def markers(track: str) -> tuple[str, str]:
    return (
        f"<!-- discophon-leaderboard:{track}:start -->",
        f"<!-- discophon-leaderboard:{track}:end -->",
    )


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


def _base_frame(scores_dir: Path, track: dict[str, Any]) -> pl.DataFrame:
    df = read_scores(scores_dir).filter(
        pl.col("split") == "test",
        pl.col("metric").is_in(track["metrics"]),
        pl.col("model").is_in(list(MODELS)),
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
    # Only compute an average for (model, layer, duration, test_split) groups that cover
    # every expected language for that split. Partial models get no avg row so their
    # average cell stays empty in the table.
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


def check_language_coverage(scores_dir: Path) -> None:
    """Raise ValueError if any non-partial MODELS entry is missing languages for any split."""
    expected = _expected_lang_counts()
    full_models = {k: v for k, v in MODELS.items() if not v.get("allow_partial")}
    if not full_models:
        return
    df = read_scores(scores_dir).filter(
        pl.col("split") == "test",
        pl.col("model").is_in(list(full_models)),
    )
    for key in full_models:
        for split_key, n_expected in expected.items():
            n_langs = df.filter((pl.col("model") == key) & (pl.col("test_split") == split_key))["language"].n_unique()
            if n_langs < n_expected:
                raise ValueError(
                    f"Model {key!r} covers only {n_langs}/{n_expected} {split_key} languages. "
                    f"Add allow_partial=True to its MODELS entry to permit partial coverage."
                )


def build_payload(scores_dir: Path, track: dict[str, Any]) -> dict[str, Any]:
    metrics = track["metrics"]
    index = _language_index()
    wide = _rows_with_avg(_base_frame(scores_dir, track)).pivot(
        on="metric",
        index=["model", "layer", "duration", "test_split", "language"],
        values=["score", "top", "std"],
    )
    rows = [_row_payload(r, metrics) for r in wide.iter_rows(named=True)]
    return {
        "metrics": [{"key": k, **METRICS[k]} for k in metrics],
        # "href"/"external" tell the page where each model name links (see model_link):
        # baselines to their HuggingFace checkpoint, submissions to an in-page section.
        "models": [
            {"key": k, "label": v["label"], "category": v["category"], **model_link(k, v)} for k, v in MODELS.items()
        ],
        "categories": CATEGORIES,
        "durations": DURATIONS,
        "splits": _splits_payload(index),
        "avg_key": AVG_LANGUAGE,
        "rows": rows,
    }


def build_track(
    track: str,
    scores_dir: Path,
    page_path: Path,
    templates_dir: Path,
    *,
    default_split: str = "test",
) -> Path:
    check_language_coverage(scores_dir)
    payload = build_payload(scores_dir, TRACKS[track])
    env = Environment(loader=FileSystemLoader(templates_dir), autoescape=select_autoescape(["html", "j2"]))
    snippet_tpl = env.get_template("leaderboard.snippet.html.j2")
    element_id = f"discophon-{track.replace('_', '-')}"
    # The snippet stays free of inline CSS/JS so the site loads them as persistent
    # assets (instant-navigation safe); see extra_css/extra_javascript in zensical.toml.
    # The payload rides in a data attribute (autoescaped by Jinja) rather than a
    # <script type="application/json">: zensical's instant navigation re-executes inline
    # scripts on page swap, which would try to run the JSON as JS and drop it.
    snippet = snippet_tpl.render(
        metrics=payload["metrics"],
        data_json=json.dumps(payload),
        default_split=default_split,
        element_id=element_id,
    )
    # Inject the leaderboard HTML between the markers, leaving the rest of the page
    # authored by hand. Raw HTML in markdown needs no extension, which lets us avoid
    # pymdownx.snippets (it would clobber zensical's defaults; see zensical.toml).
    marker_start, marker_end = markers(track)
    page = page_path.read_text(encoding="utf-8")
    start, end = page.find(marker_start), page.find(marker_end)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"{page_path} must contain '{marker_start}' ... '{marker_end}' markers")
    updated = page[: start + len(marker_start)] + f"\n{snippet}\n" + page[end:]
    page_path.write_text(updated, encoding="utf-8")
    return page_path


def build_topline_payload(scores_dir: Path) -> dict[str, Any]:
    track_metrics: list[str] = TRACKS["many_to_one"]["metrics"]  # ty:ignore[invalid-assignment]
    index = _language_index()
    base = (
        read_scores(scores_dir)
        .filter(
            pl.col("split") == "test",
            pl.col("metric").is_in(track_metrics),
            pl.col("model").is_in(list(TOPLINE_MODELS)),
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
            {"key": k, "label": v["label"], "category": "topline", "href": v["href"], "external": True}
            for k, v in TOPLINE_MODELS.items()
        ],
        "categories": [{"key": "topline", "label": "Toplines"}],
        "durations": [{"key": "0", "label": "Zero-shot"}],
        "splits": _splits_payload(index),
        "avg_key": AVG_LANGUAGE,
        "rows": rows,
    }


def build_topline(scores_dir: Path, page_path: Path, templates_dir: Path, *, default_split: str = "test") -> Path:
    payload = build_topline_payload(scores_dir)
    env = Environment(loader=FileSystemLoader(templates_dir), autoescape=select_autoescape(["html", "j2"]))
    snippet_tpl = env.get_template("leaderboard.snippet.html.j2")
    snippet = snippet_tpl.render(
        metrics=payload["metrics"],
        data_json=json.dumps(payload),
        default_split=default_split,
        element_id="discophon-topline",
    )
    marker_start, marker_end = markers("topline")
    page = page_path.read_text(encoding="utf-8")
    start, end = page.find(marker_start), page.find(marker_end)
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"{page_path} must contain '{marker_start}' ... '{marker_end}' markers")
    updated = page[: start + len(marker_start)] + f"\n{snippet}\n" + page[end:]
    page_path.write_text(updated, encoding="utf-8")
    return page_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scores", type=Path, help="directory of *.jsonl score files")
    parser.add_argument("templates", type=Path, help="templates directory")
    parser.add_argument("page", type=Path, help="markdown page to inject the leaderboard into (between markers)")
    parser.add_argument("--track", choices=list(TRACKS), default="many_to_one")
    parser.add_argument(
        "--default-split",
        choices=("dev", "test"),
        default="test",
        help="initial split shown in the HTML page",
    )
    parser.add_argument("--topline", action="store_true", help="build the interactive topline table instead")
    args = parser.parse_args()
    if args.topline:
        page_path = build_topline(args.scores, args.page, args.templates, default_split=args.default_split)
    else:
        page_path = build_track(args.track, args.scores, args.page, args.templates, default_split=args.default_split)
    print(f"page: {page_path}")


if __name__ == "__main__":
    main()
