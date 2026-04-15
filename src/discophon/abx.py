"""ABX discriminability.

We split this part of the evaluation in a separate module because it's optional
and takes more time to compute. If you want to use it, install `fastabx` either
with `pip install discophon[abx]` or `pip install fastabx`.
"""

from pathlib import Path
from typing import Literal, TypedDict, overload

try:
    from fastabx import Dataset, Score, Subsampler, Task
    from fastabx.distance import DistanceName
    from fastabx.zerospeech import InvalidSpeakerOrContextError
except ImportError as error:
    raise ImportError(
        "fastabx is required for ABX evaluation. "
        "Please install it with `pip install discophon[abx]` or `pip install fastabx`."
    ) from error

__all__ = ["continuous_abx", "discrete_abx"]


class TriphoneABX(TypedDict):
    within_speaker: float
    across_speaker: float


class PhonemeABX(TypedDict):
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
    subsampler = Subsampler(max_size_group=500 if context == "any" else None, max_x_across=5, seed=seed)
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
    kind: Literal["triphone", "phoneme"] = "triphone",
) -> TriphoneABX | PhonemeABX:
    """ABX on discrete units.

    Arguments:
        path_item: Path to the ABX item file
        path_units: Path to the predicted units: JSONL file with keys `file` ([`str`][]) and `units` (`list[int]`).
        frequency: Feature frequency in Hz. It is the inverse of the `step_units` parameter used in other functions.
        kind: Kind of representations to consider. If `phoneme`, we also compute the ABX in the "any" context
              condition, if addition of "within" context.

    Returns:
        Dictionary of ABX discriminabilities with keys `"within_speaker"` and `"across_speaker"` if `kind` is
            `"phoneme"`, and with keys `"within_speaker_within_context"`, `"across_speaker_within_context"`,
            `"within_speaker_any_context"`, and `"across_speaker_any_context"` otherwise.
    """
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
    kind: Literal["triphone", "phoneme"] = "triphone",
) -> TriphoneABX | PhonemeABX:
    """ABX on continuous representations.

    Arguments:
        path_item: Path to the ABX item file
        path_features: Path to the extracted features: folder of `.pt` files with names corresponding to the file ids.
        frequency: Feature frequency in Hz. It is the inverse of the `step_units` parameter used in other functions.
        kind: Kind of representations to consider. If `phoneme`, we also compute the ABX in the "any" context
              condition, if addition of "within" context.

    Returns:
        Dictionary of ABX discriminabilities with keys `"within_speaker"` and `"across_speaker"` if `kind` is
            `"phoneme"`, and with keys `"within_speaker_within_context"`, `"across_speaker_within_context"`,
            `"within_speaker_any_context"`, and `"across_speaker_any_context"` otherwise.
    """
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

    parser = argparse.ArgumentParser(
        prog="discophon.evaluate.abx",
        description="Continuous or discrete ABX",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("item", type=Path, help="Path to the item file")
    parser.add_argument(
        "root",
        type=Path,
        help="Path to the JSONL with units or directory with continuous features",
    )
    parser.add_argument("--frequency", required=True, type=float, help="Required. Units frequency in Hz")
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
