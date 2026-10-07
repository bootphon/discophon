import math
import os


def split_across_slurm_array(n_total: int) -> tuple[int, int]:
    """Range of indices `[start, end[` to process by the current SLURM array task, out of `n_total`."""
    if "SLURM_NTASKS" not in os.environ:
        return 0, n_total
    if "SLURM_ARRAY_TASK_ID" in os.environ:
        array_id, num_arrays = int(os.environ["SLURM_ARRAY_TASK_ID"]), int(os.environ["SLURM_ARRAY_TASK_COUNT"])
        if os.environ["SLURM_ARRAY_TASK_MIN"] != "0" or int(os.environ["SLURM_ARRAY_TASK_MAX"]) != num_arrays - 1:
            raise ValueError(
                f"Inside a SLURM array, but {os.environ['SLURM_ARRAY_TASK_MIN']=} and "
                f"{os.environ['SLURM_ARRAY_TASK_MAX']=} are not consistent with "
                f"{os.environ['SLURM_ARRAY_TASK_COUNT']=}."
            )
    else:
        array_id, num_arrays = 0, 1
    n_per_array = math.ceil(n_total / num_arrays)
    start = array_id * n_per_array
    end = min(start + n_per_array, n_total)
    return start, end
