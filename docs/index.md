<!--
title: DiscoPhon
template: main.html
-->

![DiscoPhon](https://raw.githubusercontent.com/bootphon/discophon/refs/heads/main/docs/assets/logo-full.svg)

<div class="page-subtitle">Benchmarking the Unsupervised Discovery of Phoneme Inventories With Discrete Speech Units</div>

[arXiv](https://arxiv.org/abs/2603.18612) · [DOI](https://www.isca-archive.org/interspeech_2026/poli26_interspeech.html) · [GitHub](https://github.com/bootphon/discophon) · [Website](https://benchmarks.cognitive-ml.fr/discophon)

DiscoPhon is a multilingual benchmark evaluating unsupervised phoneme discovery from discrete speech units.
Given only 10 hours of speech in an unseen language, models must produce discrete units that map to a predefined phoneme inventory.

## Getting started

DiscoPhon requires **Python ≥ 3.12**.

- Install this package:
  ```bash
  pip install discophon              # core: phoneme discovery evaluation
  pip install "discophon[prepare]"   # adds the data download and preparation
  pip install "discophon[abx]"       # adds ABX discriminability (fastabx)
  pip install "discophon[baselines]" # adds the baseline models
  ```
  Only the `baselines` extra has a system dependency:
  [FFmpeg](https://ffmpeg.org/download.html), required by `torchcodec` to read audio.
- [Follow the tutorials](https://benchmarks.cognitive-ml.fr/discophon/guide/) to download data, evaluate models, and prepare your submission.
- [Current leaderboard](https://benchmarks.cognitive-ml.fr/discophon/leaderboard/).

## References

```bibtex
@inproceedings{poli2026discophon,
  title     = {{DiscoPhon: Benchmarking the Unsupervised Discovery of Phoneme Inventories With Discrete Speech Units}},
  author    = {Maxime Poli and Manel Khentout and Angelo {Ortiz Tandazo} and Ewan Dunbar and Emmanuel Chemla and Emmanuel Dupoux},
  year      = {2026},
  booktitle = {{Interspeech 2026}},
  pages     = {6664--6669},
  doi       = {10.21437/Interspeech.2026-2791},
  issn      = {2958-1796},
}
```
