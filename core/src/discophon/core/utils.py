import math
import os
from collections.abc import Sequence


def split_for_distributed[T](sequence: Sequence[T]) -> Sequence[T]:
    if "SLURM_NTASKS" not in os.environ:
        return sequence
    rank, world_size = int(os.environ["SLURM_PROCID"]), int(os.environ["SLURM_NTASKS"])
    array_id, num_arrays = int(os.getenv("SLURM_ARRAY_TASK_ID", "0")), int(os.getenv("SLURM_ARRAY_TASK_COUNT", "1"))
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        assert os.environ["SLURM_ARRAY_TASK_MIN"] == "0"
        assert int(os.environ["SLURM_ARRAY_TASK_MAX"]) == num_arrays - 1

    n_total = len(sequence)  # Split by array first
    files_per_array = math.ceil(n_total / num_arrays)
    start = array_id * files_per_array
    end = min(start + files_per_array, n_total)
    sequence = sequence[start:end]

    n_local = len(sequence)  # Then split by rank within each array
    files_per_rank = math.ceil(n_local / world_size)
    start = rank * files_per_rank
    end = min(start + files_per_rank, n_local)
    return sequence[start:end]
