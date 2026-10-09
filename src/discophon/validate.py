"""Validation of the inputs and of the dataset structure."""

from collections.abc import Callable, Sequence
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

    def __init__(
        self, first: object = None, second: object = None, names: Sequence[str] = ("first", "second")
    ) -> None:
        message = "The first two arguments must be dictionaries with the same keys"
        if isinstance(first, dict) and isinstance(second, dict):
            for name, only in [(names[0], first.keys() - second.keys()), (names[1], second.keys() - first.keys())]:
                message += f". {len(only)} keys only in `{name}`, such as {sorted(map(str, only))[:5]}"
        super().__init__(message)


def validate_first_two_arguments_same_keys[R, **P](func: Callable[P, R]) -> Callable[P, R]:
    """Check that the first two arguments of the function are dictionaries with the same keys."""
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
            raise ValidateSameKeysError(first, second, names)
        return func(*args, **kwargs)

    return wrapper


class DatasetError(ValueError):
    """Raised when the structure is wrong."""

    def __init__(self, directory: Path, found: set[str], expected: set[str], not_dirs: Sequence[str] = ()) -> None:
        details = [
            f"{label} {sorted(names)}"
            for label, names in [
                ("missing", expected - found),
                ("unexpected", found - expected),
                ("not directories", set(not_dirs)),
            ]
            if names
        ]
        super().__init__(f"Invalid discophon dataset structure in {directory}: {', '.join(details)}.")


def validate_dataset_structure(path: str | Path) -> None:
    root = Path(path).resolve()
    visible = "[!.]*"  # Ignore hidden files such as .DS_Store
    languages = all_languages()
    found, expected = {p.name for p in root.glob(visible)}, {"alignment", "audio", "item", "manifest"}
    if found != expected:
        raise DatasetError(root, found, expected)
    found = {p.name for p in (root / "alignment").glob(visible)}
    expected = set(starmap(alignment_filename, product(languages, ["dev", "test"])))
    if found != expected:
        raise DatasetError(root / "alignment", found, expected)
    found = {p.name for p in (root / "item").glob(visible)}
    expected = {
        item_filename(lang, split, kind=kind)
        for kind, lang, split in product(["triphone", "phoneme"], languages, ["dev", "test"])
    }
    if found != expected:
        raise DatasetError(root / "item", found, expected)
    found = {p.name for p in (root / "manifest").glob(visible)}
    expected = set(
        starmap(manifest_filename, product(languages, ["dev", "test", "train-10h", "train-10min", "train-1h"]))
    ) | {"speakers.jsonl"}
    if found != expected:
        raise DatasetError(root / "manifest", found, expected)
    audio_languages = list((root / "audio").glob(visible))
    found, expected = {p.name for p in audio_languages}, {lang.iso_639_3 for lang in languages}
    not_dirs = [p.name for p in audio_languages if not p.is_dir()]
    if found != expected or not_dirs:
        raise DatasetError(root / "audio", found, expected, not_dirs)
    expected = {"all", "dev", "test", "train-10h", "train-10min", "train-1h"}
    for lang in languages:
        audio_splits = list((root / "audio" / lang.iso_639_3).glob(visible))
        found = {p.name for p in audio_splits}
        not_dirs = [p.name for p in audio_splits if not p.is_dir()]
        if found != expected or not_dirs:
            raise DatasetError(root / "audio" / lang.iso_639_3, found, expected, not_dirs)


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
