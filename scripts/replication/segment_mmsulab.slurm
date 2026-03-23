#!/bin/bash
#SBATCH -J segment
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=10:00:00
#SBATCH --array=0-255

set -euo pipefail
srun python segment_mmsulab.py "$WORK/data/mms_ulab_v2" "$SCRATCH/mmsulab" --rttm "./mms_ulab_v2.rttm"
