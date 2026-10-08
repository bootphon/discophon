import json
from pathlib import Path

import polars as pl
import pytest

from discophon.languages import all_languages
from discophon.leaderboard import TRACKS, build, cli, export, read_leaderboard_scores, read_models, validate

ROOT = Path(__file__).parent.parent / "leaderboard"
MODEL = """
[my-model]
label = "My Model"
category = "submission"
url = "https://example.com"
"""


DISCOVERY = ["per", "r_val", "f1", "pnmi"]
CONTINUOUS = ["triphone_abx_continuous_within_speaker", "triphone_abx_continuous_across_speaker"]


def write_scores(path: Path, metrics: list[str], score: float) -> None:
    rows = [
        {"language": lang.iso_639_3, "split": split, "metric": metric, "score": score}
        for lang in all_languages()
        for split in ["dev", "test"]
        for metric in metrics
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows).write_ndjson(path)


def artifacts(model_dir: Path, layers: dict[str, dict[str, int]]) -> Path:
    """Model directory with scores for layers 1 and 2 (score = layer / 10) in every condition and folder."""
    conditions = ["zero-shot", *(f"ft-{lang.iso_639_3}-10h" for lang in all_languages())]
    for condition in conditions:
        for layer in [1, 2]:
            for folder, metrics in [("many_to_one", DISCOVERY), ("one_to_one", DISCOVERY), ("continuous", CONTINUOUS)]:
                write_scores(model_dir / condition / folder / str(layer) / "scores.jsonl", metrics, layer / 10)
    (model_dir / "info.json").write_text(json.dumps({"step_units": 20, "layers": layers}))
    return model_dir


def test_repository_leaderboard_is_valid(tmp_path: Path) -> None:
    build(ROOT, tmp_path / "data.js")
    assert (tmp_path / "data.js").read_text().startswith("window.DISCOPHON_LEADERBOARD = {")


def test_export(tmp_path: Path) -> None:
    layers = {"many_to_one": {"0": 2, "10h": 1}, "one_to_one": {"0": 1}}
    exported = export(artifacts(tmp_path / "my-model", layers))
    many_to_one, one_to_one = exported["many_to_one"], exported["one_to_one"]
    assert dict(many_to_one.select("duration", "layer").unique().iter_rows()) == {"0": 2, "10h": 1}
    assert set(many_to_one["metric"]) == set(TRACKS["many_to_one"])
    assert len(many_to_one) == 2 * len(all_languages()) * len(TRACKS["many_to_one"])  # finetuned: own language only
    assert set(many_to_one.filter(pl.col("layer") == 2)["score"]) == {20.0}
    assert set(one_to_one["duration"]) == {"0"}
    assert set(one_to_one["metric"]) == set(TRACKS["one_to_one"])


@pytest.mark.parametrize(
    "info",
    [
        {"layers": {"many_to_one": {"0": 1}}},
        {"step_units": 20, "layers": {"many_to_one-k1024": {"0": 1}}},
        {"step_units": 20, "layers": {"many_to_one": {"1h": 1}}},
        {"step_units": 20, "layers": {"many_to_one": {}}},
    ],
)
def test_invalid_info(tmp_path: Path, info: dict) -> None:
    (tmp_path / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match=r"info\.json"):
        export(tmp_path)


def test_export_missing_scores(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="No scores"):
        export(artifacts(tmp_path / "my-model", {"many_to_one": {"0": 3}}))


def test_export_then_build(tmp_path: Path) -> None:
    root = tmp_path / "leaderboard"
    root.mkdir()
    (root / "models.toml").write_text(MODEL)
    model_dir = artifacts(tmp_path / "my-model", {"many_to_one": {"0": 2, "10h": 2}, "one_to_one": {"0": 1, "10h": 2}})
    for _ in range(2):  # exporting again overwrites
        cli(["export", str(model_dir), "--root", str(root)])
    cli(["build", "--root", str(root), "--output", str(tmp_path / "data.js")])
    models = read_models(root)
    scores = read_leaderboard_scores(root, models)
    validate(scores, models)
    layers = scores.filter(track="one_to_one").select("duration", "layer").unique().iter_rows()
    assert dict(layers) == {"0": 1, "10h": 2}


@pytest.mark.parametrize(
    "entry",
    [
        '[My_Model]\nlabel = "x"\ncategory = "submission"\nurl = "x"',
        '[my-model]\nlabel = "x"\ncategory = "submission"',
        '[my-model]\nlabel = "x"\ncategory = "other"\nurl = "x"',
        '[my-model]\nlabel = "x"\ncategory = "submission"\nurl = "x"\nauthors = "x"',
        '[my-model]\nlabel = "x"\ncategory = "submission"\nurl = "x"\npartial = "yes"',
    ],
)
def test_invalid_registry(tmp_path: Path, entry: str) -> None:
    (tmp_path / "models.toml").write_text(entry)
    with pytest.raises((ValueError, TypeError)):
        read_models(tmp_path)


def test_invalid_scores(tmp_path: Path) -> None:
    model_dir = artifacts(tmp_path / "my-model", {"many_to_one": {"0": 2, "10h": 2}})
    df = export(model_dir)["many_to_one"].with_columns(model=pl.lit("my-model"), track=pl.lit("many_to_one"))
    models = {"my-model": {"label": "x", "category": "submission", "url": "x"}}
    validate(df, models)
    with pytest.raises(ValueError, match="Duplicate"):
        validate(pl.concat([df, df.head(1)]), models)
    with pytest.raises(ValueError, match="cover every language"):
        validate(df.slice(1), models)
    validate(df.slice(1), {"my-model": models["my-model"] | {"partial": True}})
    with pytest.raises(ValueError, match="same layer"):
        validate(df.with_columns(layer=pl.int_range(pl.len())), models)
    with pytest.raises(ValueError, match="invalid metrics"):
        validate(df.with_columns(track=pl.lit("one_to_one")), models)
