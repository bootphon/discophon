# The Phoneme Discovery benchmark

[💾 [Website](https://benchmarks.cognitive-ml.fr/phoneme_discovery)] [📜 [Paper]()] [📖 [BibTex](https://github.com/bootphon/phoneme_discovery?tab=readme-ov-file#citation)]

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

## Installation

```bash
pip install discophon
```

To be able to compute ABX discriminabilities: `pip install discophon[abx]`.

If you want to run baselines and have access to the utility scripts, clone this repository:

```bash
git clone https://github.com/bootphon/phoneme_discovery
cd phoneme_discovery
uv sync
# uv sync --all-extras --all-groups # If you want all dependencies
```

## Usage

Check out the documentation:

- [Data preparation](https://github.com/bootphon/phoneme_discovery/blob/main/docs/prepare.md)
- [Simple evaluation](https://github.com/bootphon/phoneme_discovery/blob/main/docs/evaluate.md)
- [Run the benchmark](https://github.com/bootphon/phoneme_discovery/blob/main/benchmark.md)
- [Use the baseline systems](https://github.com/bootphon/phoneme_discovery/blob/main/baselines.md)

### Citation

```bibtex

```

Contact: `benchmarks [at] cognitive-ml [dot] fr`
