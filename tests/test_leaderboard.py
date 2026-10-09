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


def read_data(path: Path) -> dict:
    text = path.read_text(encoding="utf-8")
    return json.loads(text.removeprefix("window.DISCOPHON_LEADERBOARD = ").removesuffix(";\n"))


def test_repository_leaderboard_is_valid(tmp_path: Path) -> None:
    build(ROOT, tmp_path / "data.js")
    data = read_data(tmp_path / "data.js")
    assert [model["key"] for model in data["models"]] == list(read_models(ROOT))


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
        {"step_units": 20, "layers": {"many_to_one": {"0": True}}},
    ],
)
def test_invalid_info(tmp_path: Path, info: dict) -> None:
    (tmp_path / "info.json").write_text(json.dumps(info))
    with pytest.raises(ValueError, match=r"info\.json"):
        export(tmp_path)


def test_export_from_model_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    model_dir = artifacts(tmp_path / "my-model", {"many_to_one": {"0": 2}})
    (tmp_path / "leaderboard").mkdir()
    (tmp_path / "leaderboard" / "models.toml").write_text(MODEL)
    monkeypatch.chdir(model_dir)
    cli(["export", ".", "--root", str(tmp_path / "leaderboard")])
    assert (tmp_path / "leaderboard" / "scores" / "many_to_one" / "my-model.jsonl").is_file()


@pytest.mark.parametrize("command", ["export .", "build"])
def test_cli_requires_the_leaderboard_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], command: str
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit):
        cli(command.split())
    assert "run this from the root of the repository" in capsys.readouterr().err
    assert not any(tmp_path.iterdir())


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
    data = read_data(tmp_path / "data.js")
    assert data["models"][0]["tracks"] == ["many_to_one", "one_to_one"]
    assert len(data["rows"]) == 2 * len(all_languages()) * sum(len(metrics) for metrics in TRACKS.values())
    models = read_models(root)
    scores = read_leaderboard_scores(root, models)
    validate(scores, models)
    layers = scores.filter(track="one_to_one").select("duration", "layer").unique().iter_rows()
    assert dict(layers) == {"0": 1, "10h": 2}


@pytest.mark.parametrize(
    ("entry", "match"),
    [
        ('[My_Model]\nlabel = "x"\ncategory = "submission"\nurl = "https://x"', "keys must be"),
        ('[my-model]\nlabel = "x"\ncategory = "submission"', "missing fields"),
        ('[my-model]\nlabel = "x"\ncategory = "other"\nurl = "https://x"', "category"),
        ('[my-model]\nlabel = "x"\ncategory = "submission"\nurl = "https://x"\nauthors = "x"', "unknown fields"),
        ('[my-model]\nlabel = "x"\ncategory = "submission"\nurl = "https://x"\npartial = "yes"', "expected bool"),
        ('[my-model]\nlabel = "x"\ncategory = "submission"\nurl = "javascript:alert(1)"', "url"),
    ],
)
def test_invalid_registry(tmp_path: Path, entry: str, match: str) -> None:
    (tmp_path / "models.toml").write_text(entry)
    with pytest.raises((ValueError, TypeError), match=match):
        read_models(tmp_path)


def test_registry_and_scores_mismatch(tmp_path: Path) -> None:
    (tmp_path / "models.toml").write_text(MODEL)
    with pytest.raises(ValueError, match=r"Models without scores: \['my-model'\]"):
        read_leaderboard_scores(tmp_path, read_models(tmp_path))
    (tmp_path / "scores" / "many_to_one").mkdir(parents=True)
    (tmp_path / "scores" / "many_to_one" / "other-model.jsonl").touch()
    with pytest.raises(ValueError, match=r"Scores without models: \['other-model'\]"):
        read_leaderboard_scores(tmp_path, read_models(tmp_path))


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
    with pytest.raises(ValueError, match="Invalid track"):
        validate(df.with_columns(track=pl.lit("continuous")), models)
    with pytest.raises(ValueError, match="Invalid duration"):
        validate(df.with_columns(duration=pl.lit("1h")), models)
    with pytest.raises(ValueError, match="Invalid language"):
        validate(df.with_columns(language=pl.lit("xxx")), models)
    with pytest.raises(ValueError, match="null values"):
        validate(df.with_columns(score=None), models)
    for score in [float("nan"), float("inf"), float("-inf")]:
        with pytest.raises(ValueError, match="non-finite scores"):
            validate(df.with_columns(score=pl.lit(score)), models)
    for metric, score in [
        ("per", -1.0),
        ("r_val", 101.0),
        ("f1", -1.0),
        ("pnmi", 100.5),
        ("triphone_abx_continuous", -0.1),
    ]:
        with pytest.raises(ValueError, match=rf"{metric} scores must be in"):
            validate(df.with_columns(score=pl.when(pl.col("metric") == metric).then(score).otherwise("score")), models)
    validate(df.with_columns(score=pl.when(pl.col("metric") == "per").then(150.0).otherwise("score")), models)
    validate(df.with_columns(score=pl.when(pl.col("metric") == "r_val").then(-50.0).otherwise("score")), models)
