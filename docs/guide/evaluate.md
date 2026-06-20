# Evaluation

There are two interfaces to evaluate your predicted units:

- High level with [`discophon.benchmark`][discophon.benchmark], which will run the complete evaluation suite on
  all available units.
- Low level with [`discophon.evaluate`][discophon.evaluate], where you have fine-grain control over the metrics.

## Resolution

All evaluation runs at a fixed frame rate. By default, units are assumed to be at **50 Hz, i.e. one
unit every 20 ms**. This is controlled by `step_units` (the `--step-units` flag on the CLI), with
`frequency = 1000 // step_units`. The gold phone annotations are resampled to match this resolution
automatically.

If your model emits frames at a different rate, you **must** set `step_units` accordingly. Otherwise
your units and the gold phones are misaligned and every metric is wrong:

| Model frame rate | `step_units` | `frequency` |
|------------------|--------------|-------------|
| 50 Hz (20 ms)    | 20 (default) | 50          |
| 100 Hz (10 ms)   | 10           | 100         |
| 40 Hz (25 ms)    | 25           | 40          |

Each unit (or feature frame) is taken to span exactly `step_units` ms, in order, starting from the
beginning of the file.

## High level

To run the complete benchmark evaluation, you first need to save your predicted units to JSONL files organized like this:

```console
units/
├── units-cmn-dev.jsonl
├── units-cmn-test.jsonl
├── units-deu-dev.jsonl
├── ...
├── units-wol-dev.jsonl
└── units-wol-test.jsonl
```

The filenames should be in the format `units-{language}-{split}.jsonl`, where `language` is the language code[^1],
and `split` is the dataset split (`test` or `dev`).

Each line of a JSONL file is a JSON object with exactly two fields: `file`, the audio file id (a `str`), and
`units`, the sequence of discrete units for that file (a `list[int]`):

```json
{"file": "0188-135249-0001", "units": [12, 12, 45, 3, 3, 3, 78]}
```

### Producing units from your own model

You are responsible for turning your model's output into these JSONL files. For each language and
split, iterate over the audio files of that split and write one line per file. After
[data preparation](./prepare.md), the per-split audio lives under `$DATA/audio/{code}/{split}/`:

```python
import json
from pathlib import Path

data, code, split = Path("/path/to/discophon_data"), "eng", "test"
with open(f"units/units-{code}-{split}.jsonl", "w") as f:
    for wav in sorted((data / "audio" / code / split).glob("*.wav")):
        units = my_model(wav)  # your model: -> list[int], one unit per frame
        f.write(json.dumps({"file": wav.stem, "units": units}) + "\n")
```

The `file` id is the stem of the audio file, without extension or directory (e.g.
`0188-135249-0001`). Make sure the units match the resolution described in
[Resolution](#resolution) above.

[^1]:
    dev languages: `deu`, `swa`, `tam`, `tha`, `tur`, `ukr`.

    test languages: `cmn`, `eng`, `eus`, `fra`, `jpn`, `wol`. 

You can run the benchmark evaluation on phoneme discovery with a many-to-one mapping on all
available languages and splits like this:

```python
from discophon.benchmark import benchmark_discovery

df = benchmark_discovery("/path/to/discophon_data", "/path/to/units", kind="many-to-one")
print(df)  # pl.DataFrame with the results for each language and split
```

The result is a long-format DataFrame with columns `language`, `split`, `metric`, and `score`. The
`metric` column holds one row per metric and language/split: `pnmi`, `per` (phone error rate), and
the segmentation scores `f1` and `r_val` ($R$-value). The ABX helpers return DataFrames with the
same layout.

Use the functions [`benchmark_abx_continuous`][discophon.benchmark.benchmark_abx_continuous] or
[`benchmark_abx_discrete`][discophon.benchmark.benchmark_abx_discrete] for ABX evaluation.

Via the CLI:

```console
❯ python -m discophon.benchmark --help
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
[`Phones`][discophon.data.Phones]. You also need to set the kind of evaluation `kind`, the number of units `n_units`, and the `language` or number of phonemes
`n_phonemes`.

Example:

```python
from discophon.data import read_gold_annotations, read_submitted_units
from discophon.evaluate import phoneme_discovery

phones = read_gold_annotations("/path/to/discophon_data/alignment/alignment-eng-test.txt")
units = read_submitted_units("/path/to/units/units-eng-test.jsonl")
result = phoneme_discovery(units, phones, kind="many-to-one", n_units=256, language="eng")
print(result)
```

Or via the CLI:

```console
❯ python -m discophon.evaluate --help
usage: discophon.evaluate [-h] [--language LANGUAGE] [--n-phonemes N_PHONEMES] --n-units N_UNITS
                          [--kind {many-to-one,one-to-one}] [--step-units STEP_UNITS]
                          units phones

Evaluate predicted units on phoneme discovery

positional arguments:
  units                 Path to predicted units
  phones                Path to gold alignments

options:
  -h, --help            show this help message and exit
  --language LANGUAGE   Evaluated language. Either use this or `--n-phonemes` (default: None)
  --n-phonemes N_PHONEMES
                        Number of phonemes. Either use this or `--language` (default: None)
  --n-units N_UNITS     Required. Number of units (default: None)
  --kind {many-to-one,one-to-one}
                        Kind of assignment (either many-to-one, or one-to-one) (default: many-to-
                        one)
  --step-units STEP_UNITS
                        Step between units (in ms) (default: 20)
```

### ABX

The ABX evaluation is done separately. First, install this package with the `abx` optional dependencies:

```bash
pip install discophon[abx]
```

Discrete ABX reads the same `units-{code}-{split}.jsonl` files as above. **Continuous** ABX instead
reads extracted features stored as one tensor file per audio file: save them as `.pt` files under
`path_features/{code}/{split}/`, each named after the file id, e.g.
`features/eng/test/0188-135249-0001.pt`. Each is a 2D tensor of shape `(num_frames, feature_dim)` at
the resolution set by `frequency` (the inverse of `step_units`).

Then, either run it in Python:

```python
from discophon.abx import discrete_abx, continuous_abx

result_discrete = discrete_abx(
    "/path/to/discophon_data/item/triphone-eng-test.item",
    "/path/to/units/units-eng-test.jsonl",
    frequency=50,
)
print("Discrete: ", result_discrete)

result_continuous = continuous_abx(
    "/path/to/discophon_data/item/triphone-eng-test.item",
    "/path/to/units/units-eng-test.jsonl",
    frequency=50,
)
print("Continuous: ", result_continuous)
```

Or via the CLI:

```console
❯ python -m discophon.abx --help
usage: discophon.abx [-h] --frequency FREQUENCY [--kind {triphone,phoneme}] item root

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
