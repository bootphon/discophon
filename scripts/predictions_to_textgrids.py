import argparse
from pathlib import Path

from tqdm import tqdm

from discophon.benchmark import available_languages_and_splits_for_units
from discophon.data import (
    DEFAULT_N_UNITS,
    STEP_PHONES,
    STEP_UNITS,
    read_gold_annotations,
    read_submitted_units,
    write_textgrids,
)
from discophon.evaluate import phone_assignments
from discophon.evaluate.assignment import coocurrence_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export gold phones, units, and predicted phones to TextGrid format")
    parser.add_argument("dataset", type=Path, help="Path to the benchmark dataset")
    parser.add_argument("predictions", type=Path, help="Path to the directory with the discrete units")
    parser.add_argument("outdir", type=Path, help="Output directory")
    args = parser.parse_args()

    for language, split in tqdm(available_languages_and_splits_for_units(args.predictions)):
        if split not in {"dev", "test"}:
            continue
        code = language.iso_639_3
        units = read_submitted_units(args.predictions / f"units-{code}-{split}.jsonl")
        phones = read_gold_annotations(args.dataset / f"alignment/alignment-{code}-{split}.txt")
        coocurrence = coocurrence_matrix(units, phones, n_units=DEFAULT_N_UNITS, n_phonemes=language.n_phonemes)
        predictions = phone_assignments(units, coocurrence, kind="many-to-one")
        write_textgrids(phones, args.outdir / f"{code}/{split}", tier_name="phones", step_in_ms=STEP_PHONES)
        write_textgrids(units, args.outdir / f"{code}/{split}", tier_name="units", step_in_ms=STEP_UNITS)
        write_textgrids(predictions, args.outdir / f"{code}/{split}", tier_name="predictions", step_in_ms=STEP_UNITS)
