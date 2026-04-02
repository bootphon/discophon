#!/bin/bash
#SBATCH -J discophon-segment
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --array=0-255

set -euo pipefail
srun python segment_dataset.py "$WORK/data/mms_ulab_v2" ./mms_ulab_v2.rttm "$SCRATCH/mms_ulab_v2_segmented"
