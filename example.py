from discophon.core import read_gold_annotations, read_submitted_units
from discophon.evaluate import phoneme_discovery

phones = read_gold_annotations("/store/projects/phoneme_discovery/dev-clean.align")
units = read_submitted_units("/store/projects/phoneme_discovery/spidr-codebooks-dev-clean-5.jsonl")
result = phoneme_discovery(units, phones, n_units=256, n_phones=40, step_units=20)
print(result)
