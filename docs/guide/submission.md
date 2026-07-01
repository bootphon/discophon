# Submission

To appear on the [leaderboard](../leaderboard/index.md), open a pull request following the steps below.

## 1. Prepare your score files

Run the full benchmark on all languages and splits (see [Evaluation](./evaluate.md)).
Save the results as a JSONL file where each line records one language–split–metric combination:

```json
{"language": "eng", "split": "dev", "metric": "per", "score": 0.152}
{"language": "eng", "split": "test", "metric": "per", "score": 0.148}
```

`split` is `dev` or `test`. Required metrics depend on the track:

| Metric | Many-to-one | One-to-one |
|---|---|---|
| `pnmi`, `per`, `f1`, `r_val` | required | required |
| `triphone_abx_continuous_within_speaker` | optional | — |
| `triphone_abx_continuous_across_speaker` | optional | — |
| `triphone_abx_discrete_within_speaker` | optional | — |
| `triphone_abx_discrete_across_speaker` | optional | — |

ABX is not accepted for one-to-one submissions. If you include ABX for many-to-one,
provide it for all languages and both splits or omit it entirely.
Scores are raw values in [0, 1] — the leaderboard multiplies by 100 for display.

Name the file `{model-key}-0.jsonl` and place it under `scores/`. The model key must be
lowercase, hyphenated, and unique (e.g. `wav2vec2-large-robust`).

Then add an entry for your model in `submissions.toml` at the root of the repository:

All fields are required:

```toml
[wav2vec2-large-robust]
label       = "Wav2Vec2 Large Robust"
track       = "many-to-one"  # many-to-one | one-to-one
step_units  = 20             # --step-units value used during evaluation
url         = "https://huggingface.co/your/checkpoint"
authors     = "Author Name (Affiliation)"
year        = 2026
description = "One or two sentences about the model."
```

A model entered on both tracks is two separate entries with distinct keys, one per
track — there is no `both` value.

The leaderboard generates your model's description section on the page automatically from
this metadata (the `description` field plus authors, year, and checkpoint link), so there
is no page to edit by hand — just write a clear `description`.

## 2. Validate the submission

Run the submission test suite to check that the score file is well-formed:

```console
❯ python -m pytest tests/test_submission.py --submission scores/{model-key}-0.jsonl
```

All checks must pass.

## 3. Check the documentation

Rebuild the docs and confirm your model appears in the leaderboard:

```console
❯ just docs
❯ open site/leaderboard/index.html
```

Verify that scores look reasonable and that the model link resolves correctly.

## 4. Open a pull request

Create a PR using the **submission template**. The PR must include:

- `scores/{model-key}-0.jsonl` — the score file
- An entry in `submissions.toml` — the metadata (the description section is generated from it)
