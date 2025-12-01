from pathlib import Path
from typing import Literal

from discophon.core import read_gold_annotations, read_submitted_units
from discophon.evaluate import phoneme_discovery

from .schema import Annotations, PhonemeDiscoveryScores, Submissions


def run_benchmark(
    path_submission: str | Path,
    path_annotation: str | Path,
    path_output: str | Path,
    *,
    split: Literal["full", "dev", "test"] | None = None,
) -> None:
    context = None if split is None else {"split": split}
    submissions = Submissions.model_validate_json(Path(path_submission).read_text(encoding="utf-8"), context=context)
    annotations = Annotations.model_validate_json(Path(path_annotation).read_text(encoding="utf-8"), context=context)
    results = {"pnmi": {}, "per": {}, "f1": {}, "r_val": {}}
    for iso_code, submission in submissions.items():
        phones = read_gold_annotations(annotations[iso_code].phones, step_in_ms=annotations[iso_code].step_phones)
        units = read_submitted_units(submission.units)
        this_results = phoneme_discovery(
            units,
            phones,
            n_units=submission.n_units,
            n_phones=annotations[iso_code].n_phones,
            step_units=submission.step_units,
            step_phones=annotations[iso_code].step_phones,
        )
        for metric, score in this_results.items():
            results[metric][iso_code] = score
    Path(path_output).write_text(PhonemeDiscoveryScores.model_validate(results).model_dump_json(), encoding="utf-8")
