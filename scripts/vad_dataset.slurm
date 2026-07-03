#!/bin/bash
#SBATCH -J discophon-vad
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --array=0-255

set -euo pipefail
srun python vad_dataset.py "$WORK/data/mms_ulab_v2" "./mms_ulab_v2_raw.rttm"
