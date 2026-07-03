# Submission system design

Design for the community leaderboard submission workflow. **Status: proposed (future work).**

Artifacts to build: a validation function in `discophon`, a `submissions.toml` registry, a
`scores/` directory, a description section in the leaderboard page, and a PR template.

---

## Scope & decisions

These resolve the open questions and bound the work. Each is the minimal choice that keeps the
workflow robust; revisit only with reason.

- **One submission targets one track.** `many-to-one` and `one-to-one` score the same discovered
  units under different unit→phoneme mappings, so their metric values differ and cannot share a
  score file. A model entered on both tracks is two entries. (No `both` value.)
- **Zero-shot only.** Submissions report the zero-shot block. The fine-tuned (10 h) block stays a
  baseline-only experiment for now.
- **Validation is one library function,** exposed both as a CLI and exercised by a thin test —
  one implementation, two entry points. The CLI is discoverable for contributors and runnable in
  CI; the test keeps it in the suite. (Resolves the test-vs-CLI question: it is both, over a
  shared core.)
- **ABX = continuous, within + across speaker.** These two variants are the only ABX the
  leaderboard displays (`paper.merge_metrics` averages them into `triphone_abx_continuous`).
  Discrete ABX is not collected until the board shows it.

---

## What a submitter provides

Exactly two things, in one PR, sharing one `{model-key}`:

1. `scores/{model-key}-0.jsonl` — the scores.
2. A `[{model-key}]` block in `submissions.toml` — the metadata.

The model's description section on the leaderboard page (the `#model-{model-key}` anchor the
name links to) is generated from the metadata at build time, so the submitter does not author
it. The validator checks the two agree before a human reviews.

---

## Score file

NDJSON, one object per line, same shape as `paper/scores/`:

```json
{"language": "eng", "split": "dev",  "metric": "per",  "score": 0.152}
{"language": "eng", "split": "test", "metric": "per",  "score": 0.148}
{"language": "eng", "split": "dev",  "metric": "pnmi", "score": 0.612}
```

- `language` — ISO 639-3 code in `discophon.languages.all_languages()`.
- `split` — `dev` or `test`; both required per language/metric.
- `metric` — see the table below.
- `score` — raw float (the board multiplies by 100 for display). Ranges are metric-specific; see
  [Validation](#validation).

### Metrics

| Metric | Required | Notes |
|---|---|---|
| `per`, `r_val`, `f1`, `pnmi` | yes | the four discovery metrics |
| `triphone_abx_continuous_within_speaker` | optional¹ | many-to-one only |
| `triphone_abx_continuous_across_speaker` | optional¹ | many-to-one only |

¹ ABX is all-or-nothing: provide both variants for every language and split, or neither. It is
rejected on the one-to-one track, and an ABX-only file (no discovery metrics) is rejected on both.

Line counts: required-only = 12 languages × 2 splits × 4 metrics = **96**.
With ABX (many-to-one) = 96 + 12 × 2 × 2 = **144**.

### Naming

`scores/{model-key}-0.jsonl`. The trailing `-0` is the layer index that `paper.read_scores`
parses from the stem; submissions have no layer concept, so it is always `0`. The validator
enforces the exact stem `{model-key}-0`, so a wrong suffix fails loudly instead of being
badly ingested.

**Model key:** matches `^[a-z0-9]+(-[a-z0-9]+)*$` — lowercase, single hyphens, no leading,
trailing, or double hyphen (`read_scores` splits on `-`, and the page builds a `#model-{key}`
anchor from it). Globally unique: distinct from every other `submissions.toml` key and from every
baseline key in `leaderboard.MODELS`. Pick a name that survives a rename (no `-v1`).

Files are ~50 KB; no compression, plain `.jsonl`.

---

## Metadata (`submissions.toml`)

One file at the repo root, each section keyed by the model key (= score filename stem without `-0`):

All fields are required:

```toml
[wav2vec2-large-robust]
label       = "Wav2Vec2 Large Robust"
track       = "many-to-one"   # many-to-one | one-to-one
step_units  = 20              # --step-units used at evaluation
url         = "https://huggingface.co/org/checkpoint"
authors     = "First Last (Institution)"
year        = 2026
description = """
One or two sentences on architecture and training setup.
"""
```

TOML because it is already the project's config format (`pyproject.toml`, `zensical.toml`) and
`tomllib` is in the stdlib — nothing extra to parse or validate it. A single registry (not
per-model sidecars) keeps all metadata reviewable in one diff, makes duplicate-key detection
trivial, and avoids loose files under `scores/`. (Sidecar YAML files were considered and rejected:
they scatter context and need directory walking, and TOML multiline strings already cover the one
feature YAML would add.)

`leaderboard.py` reads this file at build time and adds each entry to `MODELS` with
`category = "submission"` — no manual edit to `MODELS` for community submissions. Baselines stay
hardcoded in `MODELS`; they are maintained by the authors and skip this workflow.

---

## Validation

A single function — the source of truth for every rule above — returns the parsed entry or raises
with a precise message. Checks:

1. **Key agreement** — stem is `{key}-0`; `key` exists in the registry, matches the key regex, and
   collides with no baseline key.
2. **Registry entry** — all fields (`label`, `track`, `step_units`, `url`, `authors`, `year`,
   `description`) present and correctly typed, no unknown fields; `track ∈ {many-to-one, one-to-one}`;
   `step_units` and `year` positive ints.
3. **Completeness** — all 96 required `(language, split, metric)` rows present, no duplicates. ABX:
   present for all languages and splits, or for none; never on one-to-one; never alone.
4. **Languages** — every `language` is in `all_languages()`.
5. **Scores** — every `score` is finite (no NaN/Inf) and within its metric's range:

   | Metric | Range |
   |---|---|
   | `pnmi`, `f1`, ABX | `[0, 1]` |
   | `per` | `≥ 0` |
   | `r_val` | `≤ 1` |

   A blanket `[0, 1]` would wrongly reject valid runs — `per` exceeds 1 with many insertions and
   `r_val` goes negative on poor segmentation — so bounds are per-metric.

Run standalone, without benchmark data:

```console
python -m discophon.submission scores/mymodel-0.jsonl   # CLI
pytest tests/test_submission.py                          # same core, in the suite
```

The test collects every `scores/*.jsonl` and validates each, so a malformed submission fails CI on
the PR.

---

## Leaderboard integration

`leaderboard.py` reads `submissions.toml`, adds each entry to the track's models as
`category = "submission"`, and ingests `scores/` alongside the baseline scores. A submission's name
links to the in-page anchor `#model-{key}`. That description section is generated from the metadata
at build time (a Jinja template rendered to markdown and injected between the page's
`{track}:descriptions` markers, like the table itself), so no hand-authored section is needed —
the `description`, `authors`, `year`, and `url` fields drive it. The generated heading carries the
`{ #model-{key} }` anchor via the `attr_list` extension.

---

## PR template (`.github/PULL_REQUEST_TEMPLATE/submission.md`)

A named template (not the default) keeps a separate template for code PRs; submitters select it in
the GitHub UI or append `?template=submission.md`.

```markdown
## Submission checklist

- [ ] `scores/{model-key}-0.jsonl` added
- [ ] `[{model-key}]` block added to `submissions.toml`
- [ ] `python -m discophon.submission scores/{model-key}-0.jsonl` passes
- [ ] `just docs` builds and the model appears in the leaderboard with its generated description

## Model

**Name:**
**Track:** many-to-one / one-to-one
**Step units:**
**Paper or checkpoint:**
**Authors:**

Brief description (architecture, training data, …).
```
