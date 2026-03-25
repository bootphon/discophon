# Evaluation

## High level

To run the complete benchmark evaluation, you first need to save your predicted units to JSONL files organized like this:

```console
units/
├── units-deu-test.jsonl
├── units-swa-test.jsonl
├── units-tam-test.jsonl
└── ...
```

The filenames should be in the format `units-{language}-{split}.jsonl`, where `language` is the language code, and `split` is the dataset split (e.g., `test` or `dev`).

Let's say you have saved the benchmark data in `./dataset`.
Then, for example, you can run the benchmark evaluation on phoneme discovery with a many-to-one mapping on all available languages and splits like this:

```python
from discophon.benchmark import benchmark_discovery

df = benchmark_discovery("./dataset", "./units", kind="many-to-one")
print(df) # pl.DataFrame with the results for each language and split
```

Use the functions [`benchmark_abx_continuous`][discophon.benchmark.benchmark_abx_continuous] or
[`benchmark_abx_discrete`][discophon.benchmark.benchmark_abx_discrete] for ABX evaluation.

Via the CLI:

```console
usage: discophon.benchmark [-h] [--benchmark {discovery,abx-discrete,abx-continuous}] [--kind {many-to-one,one-to-one}] [--step-units STEP_UNITS]
                           dataset predictions output

Phoneme Discovery benchmark

positional arguments:
  dataset               Path to the benchmark dataset
  predictions           Path to the directory with the discrete units or the features
  output                Path to the output file

options:
  -h, --help            show this help message and exit
  --benchmark {discovery,abx-discrete,abx-continuous}
                        Which benchmark (default: discovery)
  --kind {many-to-one,one-to-one}
                        Kind of assignment (either many-to-one, or one-to-one) (default: many-to-one)
  --step-units STEP_UNITS
                        Step in ms between units or features. 'frequency' is then set to 1000 // step_units. (default: 20)
```

## Low level

### Phoneme discovery

You can use the [`phoneme_discovery`][discophon.evaluate.phoneme_discovery] function with `units` of type [`Units`][discophon.data.Units], and `phones` of type
[`Phones`][discophon.data.Phones]. You also need to set the number of units `n_units`, of phonemes `n_phones`, and the step (in ms)
between consecutive units `step_units`.

Example:

```python
from discophon.data import read_gold_annotations, read_submitted_units
from discophon.evaluate import phoneme_discovery

phones = read_gold_annotations("/path/to/alignments/dataset.txt")
units = read_submitted_units("/path/to/predictions/units.jsonl")
result = phoneme_discovery(units, phones, n_units=256, n_phonemes=40, step_units=20)
print(result)
```

Or via the CLI:

```console
❯ python -m discophon.evaluate --help
usage: discophon.evaluate [-h] --n-phonemes N_PHONEMES --n-units N_UNITS [--kind {many-to-one,one-to-one}] [--step-units STEP_UNITS] units phones

Evaluate predicted units on phoneme discovery

positional arguments:
  units                 Path to predicted units
  phones                Path to gold alignments

options:
  -h, --help            show this help message and exit
  --n-phonemes N_PHONEMES
                        Required. Number of phonemes (default: None)
  --n-units N_UNITS     Required. Number of units (default: None)
  --kind {many-to-one,one-to-one}
                        Kind of assignment (either many-to-one, or one-to-one) (default: many-to-one)
  --step-units STEP_UNITS
                        Step between units (in ms) (default: 20)
```

### ABX

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
usage: discophon.evaluate.abx [-h] --frequency FREQUENCY [--kind {triphone,phoneme}] item root

Continuous or discrete ABX

positional arguments:
  item                  Path to the item file
  root                  Path to the JSONL with units or directory with continuous features

options:
  -h, --help            show this help message and exit
  --frequency FREQUENCY
                        Required. Units frequency in Hz (default: None)
  --kind {triphone,phoneme}
                        Triphone- or phoneme-based ABX (default: triphone)
```
