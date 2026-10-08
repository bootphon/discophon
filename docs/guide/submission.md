# Submission

Submitting a model to the [leaderboard](../leaderboard/index.md) takes two pull requests:

- one on the [artifacts dataset](https://huggingface.co/datasets/coml/discophon-artifacts) on the Hugging Face Hub
  with the scores and discrete units of the model, so that anyone can check and reuse them,
- one on [GitHub](https://github.com/bootphon/discophon) with the leaderboard scores, exported from the first.

Both use the same model key: lowercase letters, digits, and single hyphens (e.g. `my-model`).

## 1. Evaluate your model

There are two tracks, `many_to_one` (256 units) and `one_to_one` (as many units as phonemes, plus one), each with
two conditions:

- **Zero-shot**: the model as is.
- **Finetuned on 10h**: one model per language, finetuned on the 10h training set of that language and
  evaluated on that language only.

Choose the layer to submit using the dev languages only. You can choose a different layer for each track and
condition, as long as all the finetuned models of a track use the same layer.

Extract the units (see [Evaluate](evaluate.md)) and run the benchmark in a directory named after your model,
with one subdirectory per run, `{condition}/{track}/{layer}/`:

```
my-model/
├── info.json
├── zero-shot/
│   ├── many_to_one/6/
│   │   ├── units-{language}-{split}.jsonl
│   │   └── scores.jsonl
│   ├── one_to_one/6/
│   │   ├── units-{language}-{split}.jsonl
│   │   └── scores.jsonl
│   └── continuous/6/
│       └── scores.jsonl
└── ft-{language}-10h/
    └── ...                                 # same as zero-shot
```

The `continuous/` directory holds the continuous ABX scores, computed on features rather than units, which are
reported in the many-to-one track. The benchmark appends its results to the output file, so all the evaluations
of a run go to the same `scores.jsonl`:

```console
❯ d=my-model/zero-shot/many_to_one/6
❯ python -m discophon.benchmark data $d $d/scores.jsonl --kind many-to-one
❯ python -m discophon.benchmark data $d $d/scores.jsonl --benchmark abx-discrete
❯ python -m discophon.benchmark data features/6 my-model/zero-shot/continuous/6/scores.jsonl --benchmark abx-continuous
❯ d=my-model/zero-shot/one_to_one/6
❯ python -m discophon.benchmark data $d $d/scores.jsonl --kind one-to-one
```

Pass `--step-units` if your units are not 20 ms apart.
Finally, write `info.json` with the step between units and the submitted layers, by track and condition
(`"0"` for zero-shot, `"10h"` for finetuned on 10h):

```json
{
  "step_units": 20,
  "layers": {
    "many_to_one": {"0": 6, "10h": 6},
    "one_to_one": {"0": 6, "10h": 8}
  }
}
```

You can include more layers or conditions than the submitted ones: only those listed in `info.json` are used
by the leaderboard.

## 2. Upload the artifacts

Open a pull request on the artifacts dataset with the [`hf` CLI](https://huggingface.co/docs/huggingface_hub/guides/cli),
leaving out the lock files written by the benchmark:

```console
❯ hf upload coml/discophon-artifacts my-model/ my-model --repo-type dataset --exclude "*.lock" --create-pr
```

## 3. Export the leaderboard scores

`discophon.leaderboard export` reads the submitted layers from your model directory, and writes the leaderboard
scores to `leaderboard/scores/{track}/{key}.jsonl` for each track in `info.json`:

```console
❯ python -m discophon.leaderboard export my-model/
```

## 4. Register your model

Add an entry to `leaderboard/models.toml`:

```toml
[my-model]
label = "My Model"
category = "submission"
url = "https://huggingface.co/org/my-model"  # checkpoint or paper
description = "One or two sentences on the architecture and training data."
```

Check that the leaderboard is valid and renders as expected:

```console
❯ python -m discophon.leaderboard build
❯ uv run --group docs zensical serve
```

Then open a pull request on GitHub with `leaderboard/models.toml` and `leaderboard/scores/*/my-model.jsonl`,
and link it to the pull request on the artifacts dataset. We merge them together.
The tests check the scores: one layer per track and condition, every language and metric, no duplicates.
