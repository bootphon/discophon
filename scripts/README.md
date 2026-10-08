# Scripts

Scripts to build the DiscoPhon figures, tables, and pretraining datasets. Run them from this directory: `uv run`
picks up the `discophon-scripts` project, with the dependency group given by `--group`.

| Script                       | Purpose                                                              |
|------------------------------|----------------------------------------------------------------------|
| `interactive_plots.py`       | Interactive figures of the documentation (`docs/assets/`)            |
| `print_tables.py`            | LaTeX tables of the paper                                            |
| `predictions_to_textgrids.py`| Export gold phones, units, and predicted phones to TextGrid          |
| `vad_dataset.py`             | Voice activity detection on MMS-ulab (SLURM array)                   |
| `post_process_dataset.py`    | Clean the VAD segments                                               |
| `segment_dataset.py`         | Cut MMS-ulab into segments (SLURM array)                             |
| `slurm.py`                   | Split the work across the tasks of a SLURM array                     |

## Figures and tables

The figures and tables are computed from the scores in the
[artifacts dataset](https://huggingface.co/datasets/coml/discophon-artifacts).

```bash
uv run --group figures python interactive_plots.py /path/to/discophon_data /path/to/discophon-artifacts ../docs/assets
uv run python print_tables.py /path/to/discophon-artifacts
```

## TextGrids

Export the gold phones, the units, and the phones predicted by the many-to-one mapping, for each language and split
with units in `/path/to/units`:

```bash
uv run python predictions_to_textgrids.py /path/to/discophon_data /path/to/units /path/to/textgrids
```

## Pretraining datasets

The following explains how to reconstruct the pretraining datasets from the original sources.

### MMS-ulab

Adapt the SLURM scripts to your setup if you want to modify paths.

1. Download [espnet/mms_ulab_v2](https://huggingface.co/datasets/espnet/mms_ulab_v2) at commit
   `621586386973799a1891e76bc99c55e5e7c3a29a` (more recent commits have removed data):

   ```bash
   uvx hf download espnet/mms_ulab_v2 --repo-type=dataset --revision 621586386973799a1891e76bc99c55e5e7c3a29a --local-dir "$WORK/data/mms_ulab_v2"
   ```

2. Get access to [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and export your
   HuggingFace token to `HF_TOKEN`.
3. Install the dependencies with `uv sync --group replication`, and run Voice Activity Detection with pyannote-audio:

   ```bash
   sbatch vad_dataset.slurm
   ```

4. Post-process the segments to remove short speech segments and split by silence:

   ```bash
   uv run --group replication python post_process_dataset.py ./mms_ulab_v2_raw.rttm ./mms_ulab_v2.rttm
   ```

   Min duration of segments: 0.5s.
   Min duration of silences: 2s.
   Max duration of segments: 30s.

5. Segment the dataset given the RTTM file:

   ```bash
   sbatch segment_dataset.slurm
   ```
