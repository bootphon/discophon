#!/bin/bash
#SBATCH -J discophon
#SBATCH --cpus-per-task=10
#SBATCH --time=05:00:00
#SBATCH --array=1-12

set -euo pipefail

usage() {
    cat << EOF
Usage: $0 DATASET UNITS FEATURES OUTPUT N_UNITS [STEP_UNITS]

Arguments:
    DATASET      Path to the dataset
    UNITS        Path to the discrete units (used for benchmarks 'discovery' and 'abx-discrete')
    FEATURES     Path to the continuous features (used for benchmark 'abx-continuous')
    OUTPUT       Output JSONL file
    N_UNITS      Number of units
    STEP_UNITS   Step size for units (default: 20)

Description:
    Runs all tasks in the Phoneme Discovery benchmark, across all languages and splits combinations.
    Array tasks 1-4:  phoneme discovery
    Array tasks 5-8:  ABX discrete
    Array tasks 9-12: ABX continuous

Example:
    sbatch $0 ./path/to/dataset ./path/to/units ./path/to/features ./path/to/output 256
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
    usage
    exit 0
fi

if [[ $# -lt 5 ]]; then
    echo "Error: Missing required arguments." >&2
    usage
    exit 1
fi

DATASET=$1
UNITS=$2
FEATURES=$3
OUTPUT=$4
N_UNITS=$5
STEP_UNITS=${6:-20}

benchmark=$(( (SLURM_ARRAY_TASK_ID - 1) / 4 ))
benchmark=$(echo "discovery abx-discrete abx-continuous" | cut -d' ' -f$((benchmark + 1)))
[[ "$benchmark" == "abx-continuous" ]] && predictions="$FEATURES" || predictions="$UNITS"
task=$(( (SLURM_ARRAY_TASK_ID - 1) % 4 ))
[[ $task -lt 2 ]] && split="dev" || split="test"
[[ $((task % 2)) -eq 0 ]] && languages="dev" || languages="test"

srun python -m discophon.benchmark \
  "$DATASET" \
  "$predictions" \
  "$OUTPUT" \
  --languages "$languages" \
  --split "$split" \
  --benchmark "$benchmark" \
  --n-units "$N_UNITS" \
  --step-units "$STEP_UNITS"
