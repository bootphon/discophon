"""ABX discriminability."""

from pathlib import Path
from typing import Literal, TypedDict, overload

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


@overload
def discrete_abx(
    path_item: str | Path,
    path_units: str | Path,
    *,
    frequency: float,
    kind: Literal["triphone"],
) -> TriphoneABX: ...


@overload
def discrete_abx(
    path_item: str | Path,
    path_units: str | Path,
    *,
    frequency: float,
    kind: Literal["phoneme"],
) -> PhonemeABX: ...


def discrete_abx(
    path_item: str | Path,
    path_units: str | Path,
    *,
    frequency: float,
    kind: Literal["triphone", "phoneme"],
) -> TriphoneABX | PhonemeABX:
    """Phoneme ABX on discrete units."""
    dataset = Dataset.from_item_and_units(path_item, path_units, frequency, audio_key="file")
    match kind:
        case "triphone":
            return TriphoneABX(
                within_speaker=abx(dataset, "identical", speaker="within", context="within"),
                across_speaker=abx(dataset, "identical", speaker="across", context="within"),
            )
        case "phoneme":
            return PhonemeABX(
                within_speaker_within_context=abx(dataset, "identical", speaker="within", context="within"),
                across_speaker_within_context=abx(dataset, "identical", speaker="across", context="within"),
                within_speaker_any_context=abx(dataset, "identical", speaker="within", context="any"),
                across_speaker_any_context=abx(dataset, "identical", speaker="across", context="any"),
            )
        case _:
            raise ValueError(kind)


@overload
def continuous_abx(
    path_item: str | Path,
    path_features: str | Path,
    *,
    frequency: float,
    kind: Literal["triphone"],
) -> TriphoneABX: ...


@overload
def continuous_abx(
    path_item: str | Path,
    path_features: str | Path,
    *,
    frequency: float,
    kind: Literal["phoneme"],
) -> PhonemeABX: ...


def continuous_abx(
    path_item: str | Path,
    path_features: str | Path,
    *,
    frequency: float,
    kind: Literal["triphone", "phoneme"],
) -> TriphoneABX | PhonemeABX:
    """Phoneme ABX on continuous representations."""
    dataset = Dataset.from_item(path_item, path_features, frequency)
    match kind:
        case "triphone":
            return TriphoneABX(
                within_speaker=abx(dataset, "angular", speaker="within", context="within"),
                across_speaker=abx(dataset, "angular", speaker="across", context="within"),
            )
        case "phoneme":
            return PhonemeABX(
                within_speaker_within_context=abx(dataset, "angular", speaker="within", context="within"),
                across_speaker_within_context=abx(dataset, "angular", speaker="across", context="within"),
                within_speaker_any_context=abx(dataset, "angular", speaker="within", context="any"),
                across_speaker_any_context=abx(dataset, "angular", speaker="across", context="any"),
            )
        case _:
            raise ValueError(kind)


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
    if args.root.is_dir():
        scores = continuous_abx(args.item, args.root, frequency=args.frequency, kind=args.kind)
    elif args.root.suffix == ".jsonl":
        scores = discrete_abx(args.item, args.units, frequency=args.frequency, kind=args.kind)
    else:
        raise ValueError(args.root)
    print("\n".join(f"{key}:\t{score:.2%}" for key, score in scores.items()))
