"""ABX discriminability."""

from pathlib import Path
from typing import Literal, TypedDict

from fastabx import Dataset, Score, Subsampler, Task
from fastabx.distance import DistanceName
from fastabx.zerospeech import InvalidSpeakerOrContextError


class TriphoneABX(TypedDict):
    """Output of ABX evaluation."""

    within_speaker: float
    across_speaker: float


class PhonemeABX(TypedDict):
    """Output of ABX evaluation."""

    within_speaker_within_context: float
    across_speaker_within_context: float
    within_speaker_any_context: float
    across_speaker_any_context: float


def abx(
    dataset: Dataset,
    distance_name: DistanceName,
    *,
    speaker: Literal["within", "across"],
    context: Literal["within", "any"],
    seed: int = 0,
) -> float:
    match (speaker, context):
        case ("within", "within"):
            by, across = ["prev-phone", "next-phone", "speaker"], None
        case ("within", "any"):
            by, across = ["speaker"], None
        case ("across", "within"):
            by, across = ["prev-phone", "next-phone"], ["speaker"]
        case ("across", "any"):
            by, across = None, ["speaker"]
        case _:
            raise InvalidSpeakerOrContextError
    subsampler = Subsampler(max_size_group=None, max_x_across=5, seed=seed)
    task = Task(dataset, on="#phone", by=by, across=across, subsampler=subsampler)
    levels = ([("next-phone", "prev-phone")] if context == "within" else []) + ["speaker"]
    return Score(task, distance_name).collapse(levels=levels)


def discrete_triphone_abx(path_item: str | Path, path_units: str | Path, *, frequency: float) -> TriphoneABX:
    """Phoneme ABX on discrete units."""
    dataset = Dataset.from_item_and_units(path_item, path_units, frequency, audio_key="file")
    return {
        "within_speaker": abx(dataset, "identical", speaker="within", context="within"),
        "across_speaker": abx(dataset, "identical", speaker="across", context="within"),
    }


def continuous_triphone_abx(path_item: str | Path, path_features: str | Path, *, frequency: float) -> TriphoneABX:
    """Phoneme ABX on continuous representations."""
    dataset = Dataset.from_item(path_item, path_features, frequency)
    return {
        "within_speaker": abx(dataset, "angular", speaker="within", context="within"),
        "across_speaker": abx(dataset, "angular", speaker="across", context="within"),
    }


def discrete_phoneme_abx(path_item: str | Path, path_units: str | Path, *, frequency: float) -> PhonemeABX:
    """Phoneme ABX on continuous representations."""
    dataset = Dataset.from_item_and_units(path_item, path_units, frequency, audio_key="file")
    return {
        "within_speaker_within_context": abx(dataset, "identical", speaker="within", context="within"),
        "across_speaker_within_context": abx(dataset, "identical", speaker="across", context="within"),
        "within_speaker_any_context": abx(dataset, "identical", speaker="within", context="any"),
        "across_speaker_any_context": abx(dataset, "identical", speaker="across", context="any"),
    }


def continuous_phoneme_abx(path_item: str | Path, path_features: str | Path, *, frequency: float) -> PhonemeABX:
    """Phoneme ABX on continuous representations."""
    dataset = Dataset.from_item(path_item, path_features, frequency)
    return {
        "within_speaker_within_context": abx(dataset, "angular", speaker="within", context="within"),
        "across_speaker_within_context": abx(dataset, "angular", speaker="across", context="within"),
        "within_speaker_any_context": abx(dataset, "angular", speaker="within", context="any"),
        "across_speaker_any_context": abx(dataset, "angular", speaker="across", context="any"),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Continuous or discrete ABX")
    parser.add_argument("item", type=Path, help="Path to the item file")
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the JSONL with units or directory with continuous features",
    )
    parser.add_argument("--frequency", required=True, type=int, help="Units frequency in Hz")
    parser.add_argument(
        "--kind",
        type=str,
        choices=["triphone", "phoneme"],
        default="triphone",
        help="Triphone- or phoneme-based ABX",
    )
    args = parser.parse_args()
    if args.kind == "triphone":
        if args.root.is_dir():
            score = continuous_triphone_abx(args.item, args.root, frequency=args.frequency)
        elif args.root.suffix == ".jsonl":
            score = discrete_triphone_abx(args.item, args.units, frequency=args.frequency)
        else:
            raise ValueError(args.root)
        print(f"Within speaker:\t {score['within_speaker']:.2%}\nAcross speaker:\t {score['across_speaker']:.2%}")
    else:
        if args.root.is_dir():
            score = continuous_phoneme_abx(args.item, args.root, frequency=args.frequency)
        elif args.root.suffix == ".jsonl":
            score = discrete_phoneme_abx(args.item, args.units, frequency=args.frequency)
        else:
            raise ValueError(args.root)
        print(
            f"Within speaker, within context:\t {score['within_speaker_within_context']:.2%}\n"
            f"Across speaker, within context:\t {score['across_speaker_within_context']:.2%}\n"
            f"Within speaker, any context:\t {score['within_speaker_any_context']:.2%}\n"
            f"Across speaker, any context:\t {score['across_speaker_any_context']:.2%}\n"
        )
