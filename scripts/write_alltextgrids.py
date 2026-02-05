# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "discophon>=0.0.4",
# ]
# ///
import argparse
from pathlib import Path

from discophon.data import STEP_PHONES, read_gold_annotations, read_submitted_units, write_textgrids
from discophon.evaluate.pnmi import contingency_table, mapping_many_to_one
from discophon.languages import get_language

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("phones", type=Path, help="Path to the gold alignment")
    parser.add_argument("units", type=Path, help="Path to the predicted units")
    parser.add_argument("outdir", type=Path, help="Output directory")
    parser.add_argument("--n-units", type=int, default=256, help="Number of discrete units (default: 256)")
    parser.add_argument("--step-units", type=int, default=20, help="Step between units in ms (default: 20ms)")
    args = parser.parse_args()

    language = get_language(args.phones.stem.split("-")[1])
    units = read_submitted_units(args.units)
    phones = read_gold_annotations(args.phones)
    contingency = contingency_table(
        units,
        phones,
        n_units=args.n_units,
        n_phonemes=language.n_phonemes,
        step_units=args.step_units,
        step_phones=STEP_PHONES,
    )
    mapping = mapping_many_to_one(contingency)
    predictions = {file: [mapping[u] for u in this_units] for file, this_units in units.items()}

    write_textgrids(phones, args.outdir, tier_name="phones", step_in_ms=STEP_PHONES)
    write_textgrids(units, args.outdir, tier_name="units", step_in_ms=args.step_units)
    write_textgrids(predictions, args.outdir, tier_name="predictions", step_in_ms=args.step_units)
