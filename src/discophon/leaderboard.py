"""Leaderboard: export a model's scores, validate the registry, and build the data file for the docs.

The leaderboard lives in `leaderboard/`:

- `models.toml`: one table per model, keyed by the model key.
- `scores/{track}/{key}.jsonl`: the leaderboard scores of that model on that track, one row per
  (duration, language, metric), for the layers chosen by the model's authors.

Full scores and discrete units are distributed in the artifacts dataset on the Hugging Face Hub, with one directory
per model:

    {key}/
    ├── info.json                         # {"step_units": 20, "layers": {track: {duration: layer}}}
    └── {condition}/{folder}/{layer}/     # condition: zero-shot, ft-{lang}-{duration}
        ├── units-{lang}-{split}.jsonl    # folder: many_to_one, one_to_one, many_to_one-k{N}, continuous
        └── scores.jsonl

`export` reads the leaderboard scores from such a directory, for the layers given in `info.json`.
"""

import argparse
import json
import re
import tomllib
from pathlib import Path

import polars as pl

from discophon.data import read_scores
from discophon.languages import all_languages

TRACKS = {
    "many_to_one": ["per", "r_val", "f1", "pnmi", "triphone_abx_continuous"],
    "one_to_one": ["per", "r_val", "f1", "pnmi"],
}
METRICS = {
    "per": {"label": "PER", "lower_is_better": True},
    "r_val": {"label": "R-value", "lower_is_better": False},
    "f1": {"label": "F1", "lower_is_better": False},
    "pnmi": {"label": "PNMI", "lower_is_better": False},
    "triphone_abx_continuous": {"label": "ABX", "lower_is_better": True},
}
DURATIONS = {"0": "Zero-shot", "10h": "Finetuned on 10h"}
CATEGORIES = {"submission": "Submissions", "baseline": "Baselines", "topline": "Toplines"}
KEY_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
REQUIRED_FIELDS = {"label": str, "category": str, "url": str}
OPTIONAL_FIELDS = {"description": str, "partial": bool}
ARTIFACTS_URL = "https://huggingface.co/datasets/coml/discophon-artifacts/tree/main/{key}"
SCORES_SCHEMA = {
    "duration": pl.String,
    "layer": pl.Int64,
    "language": pl.String,
    "metric": pl.String,
    "score": pl.Float64,
}


def read_info(model_dir: Path) -> dict:
    info = json.loads((model_dir / "info.json").read_text(encoding="utf-8"))
    if info.keys() != {"step_units", "layers"} or not isinstance(info["step_units"], int):
        raise ValueError(f"{model_dir}/info.json: expected 'step_units' (int) and 'layers', got {sorted(info)}.")
    for track, layers in info["layers"].items():
        if track not in TRACKS or not layers or layers.keys() - DURATIONS.keys():
            raise ValueError(f"{model_dir}/info.json: invalid layers for {track!r}: {layers}.")
        if not all(isinstance(layer, int) for layer in layers.values()):
            raise TypeError(f"{model_dir}/info.json: layers must be integers, got {layers}.")
    return info


def export(model_dir: str | Path) -> dict[str, pl.DataFrame]:
    """Read the leaderboard scores of a model in the artifacts dataset, for each track in its `info.json`.

    Finetuned models are evaluated on their finetuning language only, and continuous ABX comes from `continuous/`.
    """
    info = read_info(Path(model_dir))
    scores = read_scores(model_dir).filter(
        pl.col("split") == "test", pl.col("ft_lang").is_null() | (pl.col("ft_lang") == pl.col("language"))
    )
    exported = {}
    for track, layers in info["layers"].items():
        chosen = pl.DataFrame({"duration": list(layers), "layer": list(layers.values())})
        df = scores.filter(pl.col("folder").is_in([track, "continuous"]), pl.col("metric").is_in(TRACKS[track])).join(
            chosen, on=["duration", "layer"]
        )
        if df.is_empty():
            raise ValueError(f"No scores for track {track!r} in {model_dir}.")
        exported[track] = (
            df.with_columns((pl.col("score") * 100).round(4))
            .select(list(SCORES_SCHEMA))
            .sort("duration", "language", "metric")
        )
    return exported


def read_models(root: Path) -> dict[str, dict]:
    with (root / "models.toml").open("rb") as f:
        models = tomllib.load(f)
    for key, entry in models.items():
        if not KEY_PATTERN.fullmatch(key):
            raise ValueError(f"[{key}]: keys must be lowercase letters, digits and single hyphens.")
        missing = REQUIRED_FIELDS.keys() - entry.keys()
        unknown = entry.keys() - REQUIRED_FIELDS.keys() - OPTIONAL_FIELDS.keys()
        if missing or unknown:
            raise ValueError(f"[{key}]: missing fields {sorted(missing)}, unknown fields {sorted(unknown)}.")
        for field, value in entry.items():
            expected = (REQUIRED_FIELDS | OPTIONAL_FIELDS)[field]
            if not isinstance(value, expected):
                raise TypeError(f"[{key}] {field}: expected {expected.__name__}, got {type(value).__name__}.")
        if entry["category"] not in CATEGORIES:
            raise ValueError(f"[{key}] category: must be one of {list(CATEGORIES)}.")
    return models


def read_leaderboard_scores(root: Path, models: dict[str, dict]) -> pl.DataFrame:
    paths = sorted((root / "scores").glob("*/*.jsonl"))
    keys = {p.stem for p in paths}
    if keys != models.keys():
        raise ValueError(
            f"Models without scores: {sorted(models.keys() - keys)}. "
            f"Scores without models: {sorted(keys - models.keys())}."
        )
    return pl.concat(
        [
            pl.read_ndjson(p, schema=SCORES_SCHEMA).with_columns(model=pl.lit(p.stem), track=pl.lit(p.parent.name))
            for p in paths
        ]
    )


def validate(scores: pl.DataFrame, models: dict[str, dict]) -> None:
    languages = {language.iso_639_3 for language in all_languages()}
    for column, allowed in [("track", TRACKS), ("duration", DURATIONS), ("language", languages)]:
        if invalid := set(scores[column].unique()) - set(allowed):
            raise ValueError(f"Invalid {column}: {sorted(invalid)}.")
    keys = ["model", "track", "duration", "language", "metric"]
    if scores.select(keys).is_duplicated().any():
        raise ValueError("Duplicate scores for the same (model, track, duration, language, metric).")
    for (model, track, duration), group in scores.group_by("model", "track", "duration"):
        where = f"{model} ({track}, duration {duration})"
        if invalid := set(group["metric"]) - set(TRACKS[track]):
            raise ValueError(f"{where}: invalid metrics {sorted(invalid)}.")
        if group["layer"].n_unique() > 1:
            raise ValueError(f"{where}: scores must all come from the same layer.")
        if group.null_count().sum_horizontal().item():
            raise ValueError(f"{where}: null values.")
        if not models[model].get("partial") and len(group) != len(languages) * len(TRACKS[track]):
            raise ValueError(f"{where}: scores must cover every language and metric (or set partial = true).")


def payload(scores: pl.DataFrame, models: dict[str, dict]) -> dict:
    return {
        "tracks": {track: [{"key": m, **METRICS[m]} for m in metrics] for track, metrics in TRACKS.items()},
        "durations": [{"key": k, "label": v} for k, v in DURATIONS.items()],
        "categories": [{"key": k, "label": v} for k, v in CATEGORIES.items()],
        "languages": [
            {"iso": lang.iso_639_3, "name": lang.name, "split": lang.split}
            for lang in sorted(all_languages(), key=lambda lang: lang.name)
        ],
        "models": [
            {
                "key": key,
                "label": entry["label"],
                "category": entry["category"],
                "url": entry["url"],
                "description": entry.get("description", ""),
                "tracks": sorted(set(scores.filter(pl.col("model") == key)["track"])),
                "artifacts": ARTIFACTS_URL.format(key=key),
            }
            for key, entry in models.items()
        ],
        "rows": scores.select("model", "track", *SCORES_SCHEMA).to_dicts(),
    }


def build(root: Path, output: Path) -> None:
    models = read_models(root)
    scores = read_leaderboard_scores(root, models)
    validate(scores, models)
    output.write_text(f"window.DISCOPHON_LEADERBOARD = {json.dumps(payload(scores, models))};\n", encoding="utf-8")


def cli(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="DiscoPhon leaderboard")
    subparsers = parser.add_subparsers(dest="command", required=True)
    parser_export = subparsers.add_parser("export", help="Export the leaderboard scores of a model")
    parser_export.add_argument("model", type=Path, help="Model directory in the artifacts dataset, named by its key")
    parser_export.add_argument("--root", type=Path, default=Path("leaderboard"))
    parser_build = subparsers.add_parser("build", help="Validate the leaderboard and build the docs data file")
    parser_build.add_argument("--root", type=Path, default=Path("leaderboard"))
    parser_build.add_argument("--output", type=Path, default=Path("docs/javascripts/leaderboard-data.js"))
    args = parser.parse_args(argv)
    if args.command == "export":
        for track, df in export(args.model).items():
            output = args.root / "scores" / track / f"{args.model.resolve().name}.jsonl"
            output.parent.mkdir(parents=True, exist_ok=True)
            df.write_ndjson(output)
    else:
        build(args.root, args.output)


if __name__ == "__main__":
    cli()
