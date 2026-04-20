from collections.abc import Callable
from functools import wraps
from itertools import product
from pathlib import Path

from discophon.languages import all_languages, get_language, Language


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
    languages = all_languages()
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
    if {p.name for p in (root / "manifest").glob("*")} != (
        {
            f"manifest-{lang.iso_639_3}-{split}.csv"
            for lang, split in product(languages, ["dev", "test", "train-10h", "train-10min", "train-1h"])
        }
        | {"speakers.jsonl"}
    ):
        raise DatasetError
    if {p.name for p in (root / "audio").glob("*")} != {lang.iso_639_3 for lang in languages}:
        raise DatasetError
    splits = {"all", "dev", "test", "train-10h", "train-10min", "train-1h"}
    for lang in languages:
        if {p.stem for p in (root / "audio" / lang.iso_639_3).glob("*")} != splits:
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
