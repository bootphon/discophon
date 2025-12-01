# The Phoneme Discovery benchmark

Contact: `benchmarks [at] cognitive-ml [dot] fr`

## Introduction

The last several years have seen revolutionary improvements in both speech processing and textual natural language
processing. In both cases, unsupervised or self-supervised pre-training has been the key to models autonomously
discovering representations that are tremendously useful for doing language tasks. Yet, central to the study of human
speech processing is the phoneme inventory, a small set of discrete units that abstract away from massive pronunciation
variability in the signal.

Discovering the correct set of phonemes for a language is crucial: encode the wrong categories, and contrasts between
words are distorted or disappear; fail to categorize at all, and contrasts between words are hidden behind semantically
irrelevant variation in the signal. While much attention has been paid to whether unsupervised speech models’
(continuous or discrete) representations are predictive of phonemes, this benchmark, for the first time, explicitly
fixes the goal of learning a discrete set of categories that are in one-to-one correspondence with the phoneme
inventory of a language.

Infants appear to learn the phoneme inventory of their language effortlessly, before they can speak. They benefit from
millions of years of evolution of the human brain and body, giving them a learning architecture that allows them to
thrive in the face of scarce and noisy language data, preparing them to learn the phoneme inventory of any human
language.

The Phoneme Discovery benchmark is aimed at building models that discover phoneme inventories across various languages,
using only small amounts of speech data, and without textual data during training.

## Data preparation

```bash
pip install discophon.prepare
```

Follow [the instructions](https://github.com/bootphon/phoneme_discovery/tree/main/prepare) to:

- Download data from Common Voice.
- Download the benchmark assets, manifests and alignments.
- Resample and convert Common Voice data to WAV.

## Evaluation

```bash
pip install discophon.evaluate
```

[Check out the README](https://github.com/bootphon/phoneme_discovery/tree/main/evaluate) to know how to evaluate
your model on this benchmark.

## Baseline systems

```bash
pip install discophon.baselines
```

[Read the documentation](https://github.com/bootphon/phoneme_discovery/tree/main/baselines) to learn how to load and
finetune baseline models.

## Submission

```bash
pip install discophon.submission
```

COMING SOON: follow [the instructions](https://github.com/bootphon/phoneme_discovery/tree/main/submission) to submit
your results to the leaderboard.

## Development

For development, create a full environment with:

```bash
uv sync --all-packages --all-extras
```

### Citation

```bibtex

```
