"""Phoneme discovery evaluation."""

from pathlib import Path

import polars as pl

from .boundaries import evaluate_boundaries
from .per import phone_error_rate
from .pnmi import evaluate_pnmi_and_predict
from .utils import DiscoveryEvaluationResult, Phones, Units


def read_submitted_units(source: str | Path) -> Units:
    """Read the units from a JSONL file. Must only have fields named 'audio' (str) and 'units' (list[int])."""
    df_units = pl.read_ndjson(source, schema={"audio": pl.String, "units": pl.List(pl.Int32)})
    units = df_units.rows_by_key("audio", named=True, unique=True).items()
    return {Path(audio).stem: row["units"] for audio, row in units}


def read_task_annotations(source: str | Path) -> Phones:
    """Read the annotations from a JSONL file. Must only have fields named 'audio' (str) and 'phones' (list[str])."""
    df_phones = pl.read_ndjson(source, schema={"audio": pl.String, "phones": pl.List(pl.String)})
    phones = df_phones.rows_by_key("audio", named=True, unique=True).items()
    return {Path(audio).stem: row["phones"] for audio, row in phones}


def discovery_evaluation(
    path_units: Path,
    path_phones: Path,
    *,
    n_units: int,
    n_phones: int,
    step_units: int,
    step_phones: int,
) -> DiscoveryEvaluationResult:
    """Full evaluation of phoneme discovery: PNMI, PER, F1 and R-value boundary detection."""
    units = read_submitted_units(path_units)
    phones = read_task_annotations(path_phones)
    pnmi, predictions = evaluate_pnmi_and_predict(
        units,
        phones,
        n_units=n_units,
        n_phones=n_phones,
        step_units=step_units,
        step_phones=step_phones,
    )
    per = phone_error_rate(predictions, phones)
    detection = evaluate_boundaries(predictions, phones, step_units=step_units, step_phones=step_phones)
    return {"pnmi": pnmi, "per": per, "f1": detection.f1, "r_val": detection.r_val}
