# Replicate DiscoPhon

## Pretraining datasets

The following explains how to reconstruct the pretraining datasets from the original sources.

### MMS-ulab

Adapt the SLURM scripts to your setup if you want to modify paths.

1. Download [espnet/mms_ulab_v2](https://huggingface.co/datasets/espnet/mms_ulab_v2) at commit `621586386973799a1891e76bc99c55e5e7c3a29a` (more recent commits have removed data):

   ```bash
   uvx hf download espnet/mms_ulab_v2  --repo-type=dataset --revision 621586386973799a1891e76bc99c55e5e7c3a29a --local-dir "$WORK/data/mms_ulab_v2"
   ```

1. Get access to [pyannote/segmentation-3.0](https://huggingface.co/pyannote/segmentation-3.0) and export your HuggingFace token to `HF_TOKEN`.
1. Run Voice Activity Detection with pyannote-audio:
   ```bash
   sbatch vad_dataset.slurm
   ```
1. Post-process the segments to remove short speech segments and split by silence:

   ```bash
   uv run post_process_dataset.py ./mms_ulab_v2_raw.rttm ./mms_ulab_v2.rttm
   ```

   Min duration of segments: 0.5s.
   Min duration of silences: 2s.
   Max duration of segments: 30s.

1. Segment the dataset given the RTTM file:

   ```bash
   sbatch segment_dataset.slurm
   ```

1. Create a HuggingFace formatted dataset from the segmented files:

   ```bash
   uv run consolidate_dataset.py "$SCRATCH/mms_ulab_v2_segmented" "$SCRATCH/mmsulab"
   ```

### VP-20

## Baselines
