from collections.abc import Callable
from functools import wraps
from itertools import product
from pathlib import Path
from typing import Literal

from .languages import dev_languages, languages_in_split, test_languages


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

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if len(args) < 2:
            raise ArgumentsError
        if not isinstance(args[0], dict) or not isinstance(args[1], dict) or set(args[0]) != set(args[1]):
            raise ValidateSameKeysError
        return func(*args, **kwargs)

    return wrapper


class DatasetError(ValueError):
    def __init__(self) -> None:
        super().__init__("Invalid phoneme_discovery dataset structure. Verify your file structure!")


def validate_dataset_structure(path: str | Path) -> None:
    root = Path(path).resolve()
    languages = dev_languages() + test_languages()
    if {p.name for p in root.glob("*")} != {"alignment", "audio", "item", "manifest"}:
        raise DatasetError
    if {p.name for p in (root / "alignment").glob("*")} != {
        f"alignment-{lang.iso_639_3}-{split}.txt" for lang, split in product(languages, ["dev", "test"])
    }:
        raise DatasetError
    if {p.name for p in (root / "item").glob("*")} != {
        f"{kind}-{lang.iso_639_3}-{split}.item"
        for kind, lang, split in product(["triphone", "phoneme"], languages, ["dev", "test"])
    }:
        raise DatasetError
    if {p.name for p in (root / "manifest").glob("*")} != {
        f"manifest-{lang.iso_639_3}-{split}.item"
        for lang, split in product(languages, ["dev", "test", "train-10h", "train-10min", "train-1h"])
    }:
        raise DatasetError
    if {p.name for p in (root / "audio").glob("*")} != {lang.iso_639_3 for lang in languages}:
        raise DatasetError
    splits = {"all", "dev", "test", "train-10h", "train-10min", "train-1h"}
    for lang in languages:
        if {p.stem for p in (root / "audio" / lang.iso_639_3).glob("*")} != splits:
            raise DatasetError


def validate_units_structure(
    path: str | Path,
    *,
    languages: Literal["dev", "test"],
    split: Literal["dev", "test"],
) -> None:
    expected = {f"units-{lang.iso_639_3}-{split}.jsonl" for lang in languages_in_split(languages)}
    found = {p.name for p in Path(path).glob("*.jsonl")}
    if not expected.issubset(found):
        raise ValueError(f"Missing units. Expected in {path}:\n{list(expected)}")


def validate_features_structure(
    path: str | Path,
    *,
    languages: Literal["dev", "test"],
    split: Literal["dev", "test"],
) -> None:
    expected = {f"{lang.iso_639_3}/{split}" for lang in languages_in_split(languages)}
    found = {str(p.relative_to(path)) for p in Path(path).glob("*/*")}
    if not expected.issubset(found):
        raise ValueError(f"Missing directories with features. Expected in {path}:\n{list(expected)}")
