from collections.abc import Callable
from functools import wraps
from inspect import signature
from itertools import product, starmap
from pathlib import Path

from discophon.data import alignment_filename, item_filename, manifest_filename
from discophon.languages import Language, all_languages, get_language


class ArgumentsError(ValueError):
    """To raise if a function does not have the correct number of arguments."""

    def __init__(self, n: int = 2) -> None:
        super().__init__(f"Function must have at least {n} positional arguments to compare.")


class ValidateSameKeysError(ValueError):
    """To be raised in the decorator below."""

    def __init__(self) -> None:
        super().__init__("The first two arguments must be dictionaries with the same keys")


def validate_first_two_arguments_same_keys[R, **P](func: Callable[P, R]) -> Callable[P, R]:
    """Decoractor that checks that the first two arguments of the function are dictionaries with the same keys."""
    sig = signature(func)
    names = list(sig.parameters)[:2]

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if len(args) >= 2:
            first, second = args[:2]
        else:
            bound = sig.bind_partial(*args, **kwargs)
            if len(names) < 2 or any(name not in bound.arguments for name in names):
                raise ArgumentsError
            first, second = (bound.arguments[name] for name in names)
        if not isinstance(first, dict) or not isinstance(second, dict) or first.keys() != second.keys():
            raise ValidateSameKeysError
        return func(*args, **kwargs)

    return wrapper


class DatasetError(ValueError):
    """Raised when the structure is wrong."""

    def __init__(self) -> None:
        super().__init__("Invalid discophon dataset structure. Verify your file structure!")


def validate_dataset_structure(path: str | Path) -> None:
    root = Path(path).resolve()
    languages = all_languages()
    if {p.name for p in root.glob("*")} != {"alignment", "audio", "item", "manifest"}:
        raise DatasetError
    if {p.name for p in (root / "alignment").glob("*")} != set(
        starmap(alignment_filename, product(languages, ["dev", "test"]))
    ):
        raise DatasetError
    if {p.name for p in (root / "item").glob("*")} != {
        item_filename(lang, split, kind=kind)
        for kind, lang, split in product(["triphone", "phoneme"], languages, ["dev", "test"])
    }:
        raise DatasetError
    if {p.name for p in (root / "manifest").glob("*")} != (
        set(starmap(manifest_filename, product(languages, ["dev", "test", "train-10h", "train-10min", "train-1h"])))
        | {"speakers.jsonl"}
    ):
        raise DatasetError
    audio_languages = list((root / "audio").glob("*"))
    if {p.name for p in audio_languages} != {lang.iso_639_3 for lang in languages} or not all(
        p.is_dir() for p in audio_languages
    ):
        raise DatasetError
    splits = {"all", "dev", "test", "train-10h", "train-10min", "train-1h"}
    for lang in languages:
        audio_splits = list((root / "audio" / lang.iso_639_3).glob("*"))
        if {p.name for p in audio_splits} != splits or not all(p.is_dir() for p in audio_splits):
            raise DatasetError


class NumberPhonemesError(ValueError):
    """To raise when there is an issue between n_phonemes and language."""


def infer_number_of_phonemes(n_phonemes: int | None, language: str | Language | None) -> int:
    if n_phonemes is not None and language is not None:
        raise NumberPhonemesError("Either specify `language` or `n_phonemes`, but not both")
    if language is None:
        if n_phonemes is None:
            raise NumberPhonemesError("You must set `language` or `n_phonemes` to get the number of target phonemes")
        return n_phonemes
    return get_language(language).n_phonemes
