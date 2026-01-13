from collections.abc import Callable
from functools import wraps
from pathlib import Path
from typing import Literal


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


def verify_dataset_structure(path: str | Path) -> None:
    pass


def verify_units_structure(
    path: str | Path,
    *,
    languages: Literal["dev", "test"] | None = None,
    split: Literal["dev", "test"] | None = None,
) -> None:
    pass


def verify_features_structure(
    path: str | Path,
    *,
    languages: Literal["dev", "test"] | None = None,
    split: Literal["dev", "test"] | None = None,
) -> None:
    pass
