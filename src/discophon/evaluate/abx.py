"""ABX discriminability."""

from pathlib import Path
from typing import TypedDict

from fastabx import Dataset, Score, Subsampler, Task
from fastabx.distance import DistanceName


class ABX(TypedDict):
    """Output of ABX evaluation."""

    within: float
    across: float


def abx(dataset: Dataset, distance_name: DistanceName, *, seed: int = 0) -> ABX:
    """Compute the phoneme ABX on the given dataset, within or across speaker."""
    within = Score(
        Task(
            dataset,
            on="#phone",
            by=["next-phone", "prev-phone", "speaker"],
            subsampler=Subsampler(max_size_group=None, max_x_across=5, seed=seed),
        ),
        distance_name,
    ).collapse(levels=[("next-phone", "prev-phone"), "speaker"])
    across = Score(
        Task(
            dataset,
            on="#phone",
            by=["next-phone", "prev-phone"],
            across=["speaker"],
            subsampler=Subsampler(max_size_group=None, max_x_across=5, seed=seed),
        ),
        distance_name,
    ).collapse(levels=[("next-phone", "prev-phone"), "speaker"])
    return {"within": within, "across": across}


def discrete_abx(path_item: str | Path, path_units: str | Path, *, frequency: float) -> ABX:
    """Phoneme ABX on discrete units."""
    return abx(Dataset.from_item_and_units(path_item, path_units, frequency, audio_key="file"), "identical")


def continuous_abx(path_item: str | Path, path_features: str | Path, *, frequency: float) -> ABX:
    """Phoneme ABX on continuous representations."""
    return abx(Dataset.from_item(path_item, path_features, frequency), "angular")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(prog="discophon.evaluate.abx", description="Continuous or discrete ABX")
    parser.add_argument("item", type=Path, help="Path to the item file")
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the JSONL with units or directory with continuous features",
    )
    parser.add_argument("--frequency", required=True, type=int, help="Units frequency in Hz")
    args = parser.parse_args()
    if args.root.is_dir():
        score = continuous_abx(args.item, args.root, frequency=args.frequency)
    elif args.root.suffix == ".jsonl":
        score = discrete_abx(args.item, args.units, frequency=args.frequency)
    else:
        raise ValueError(args.root)
    print(f"Within speaker: {score['within']:.2%}\nAcross speaker: {score['across']:.2%}")
