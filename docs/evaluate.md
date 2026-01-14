# Evaluation

## Phoneme discovery

You can use the `phoneme_discovery` function with `units: dict[str, list[int]]`, and `phones: dict[str, list[str]]`.
You also need to set the number of units `n_units`, of phonemes `n_phones`, and the step (in ms) between consecutive
units `step_units`.

Example:

```python
from discophon.core import read_gold_annotations, read_submitted_units
from discophon.evaluate import phoneme_discovery

phones = read_gold_annotations("/path/to/alignments/dataset.align")
units = read_submitted_units("/path/to/predictions/units.jsonl")
result = phoneme_discovery(units, phones, n_units=256, n_phones=40, step_units=20)
print(result)
```

Or via the CLI:

```console
❯ python -m discophon.evaluate --help
usage: discophon.evaluate [-h] [--n-units N_UNITS] [--n-phones N_PHONES] [--step-units STEP_UNITS] units phones

Evaluate predicted units on phoneme discovery

positional arguments:
  units                 path to predicted units
  phones                path to gold alignments

options:
  -h, --help            show this help message and exit
  --n-units N_UNITS     number of units
  --n-phones N_PHONES   number of phonemes
  --step-units STEP_UNITS
                        step between units (in ms)
```

## ABX

The ABX evaluation is done separately. First, install this package with the `abx` optional dependencies:

```bash
pip install discophon[abx]
```

Then, either run it in Python:

```python
from discophon.evaluate.abx import discrete_abx, continuous_abx

result_discrete = discrete_abx("/path/to/item/dataset.item", "/path/to/predictions/units.jsonl", frequency=50)
print("Discrete: ", result_discrete)

result_continuous = continuous_abx("/path/to/item/dataset.item", "/path/to/features", frequency=50)
print("Continuous: ", result_discrete)
```

Or via the CLI:

```console
❯ python -m discophon.evaluate.abx --help
usage: discophon.evaluate.abx [-h] --frequency FREQUENCY item root

Continuous or discrete ABX

positional arguments:
  item                  Path to the item file
  root                  Path to the JSONL with units or directory with continuous features

options:
  -h, --help            show this help message and exit
  --frequency FREQUENCY
                        Units frequency in Hz
```
