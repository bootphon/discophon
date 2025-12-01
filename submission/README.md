# Benchmark submission

```json
{
  "eng": {
    "units": "...",
    "n_units": 100,
    "step_units": 20
  },
  "fra": {
    "units": "...",
    "n_units": 100,
    "step_units": 20
  },
  "cmn": {
    "units": "...",
    "n_units": 100,
    "step_units": 20
  },
  "jpn": {
    "units": "...",
    "n_units": 100,
    "step_units": 20
  },
  ...
}
```

Run the evaluation:

```bash
python -m discophon.submission $SUBMISSION $ANNOTATION $OUTPUT
```
