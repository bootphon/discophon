#!/bin/bash
#SBATCH -J discophon-ft-spidr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --array=1-18

set -euo pipefail

usage() {
    cat << EOF
Usage: $0 LANGUAGES DATASET NAME PROJECT WORKDIR CHECKPOINT

Arguments:
    LANGUAGES     Languages to use (either 'dev' or 'test')
    DATASET       Path to the dataset
    NAME          Name of this run
    PROJECT       Name of the project
    WORKDIR       Path to the working directory
    CHECKPOINT    Path to the pretrained SpidR checkpoint

Description:
    Continue pretraining SpidR on data from the 'dev' or 'test' languages
    Array tasks 1-6:   train-10min
    Array tasks 7-12:  train-1h
    Array tasks 13-18: train-10h

Example:
    sbatch $0 dev ./path/to/dataset spidr-ft discophon-ft ./path/to/workdir ./path/to/pretrained/checkpoint.pt
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 6 ]]; then
    echo "Error: Missing required arguments." >&2
    usage
    exit 1
fi

LANGUAGES_SPLIT=$1
DATASET=$2
COMMON_NAME=$3
PROJECT=$4
WORKDIR=$5
CHECKPOINT=$6

if [[ "$LANGUAGES_SPLIT" == "dev" ]]; then
    languages="deu swa tam tha tur ukr"
elif [[ "$LANGUAGES_SPLIT" == "test" ]]; then
    languages="cmn eng eus fra jpn wol"
else
    echo "Error: first argument must be 'dev' or 'test'" >&2
    exit 1
fi

split=$(( (SLURM_ARRAY_TASK_ID - 1) / 6 ))
split=$(echo "train-10min train-1h train-10h" | cut -d' ' -f$((split + 1)))
lang=$(( (SLURM_ARRAY_TASK_ID - 1) % 6 ))
lang=$(echo "$languages" | cut -d' ' -f$((lang + 1)))

srun python -m discophon.baselines.finetune_spidr \
    "$COMMON_NAME-$lang-$split" \
    "$PROJECT" \
    "$WORKDIR" \
    "$CHECKPOINT" \
    "$DATASET/manifest/manifest-$lang-$split.csv"
