# API reference

This page documents the public API. To run the benchmark, start with
[`discophon.benchmark`][discophon.benchmark] (high level) or
[`discophon.evaluate`][discophon.evaluate] (low level); see the
[evaluation guide](../guide/evaluate.md) for usage.

The modules, roughly in the order you will use them:

- [`discophon.prepare`][discophon.prepare] - download and pre-process the benchmark data.
- [`discophon.benchmark`][discophon.benchmark] - run the full evaluation suite over all your units.
- [`discophon.evaluate`][discophon.evaluate] - individual metrics (PNMI, PER, segmentation).
- [`discophon.abx`][discophon.abx] - optional ABX discriminability.
- [`discophon.data`][discophon.data] - read/write units and gold annotations, and shared types.
- [`discophon.languages`][discophon.languages] - language metadata and phoneme inventories.
- [`discophon.baselines`][discophon.baselines] - finetune and extract units from the baseline models.

::: discophon.benchmark
::: discophon.evaluate
::: discophon.abx
::: discophon.prepare
::: discophon.data
::: discophon.languages
::: discophon.baselines
