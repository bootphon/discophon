"""Various utilities (mostly typing)."""

from collections.abc import Callable
from functools import wraps
from typing import TypedDict

type Units = dict[str, list[int]]
type Phones = dict[str, list[str]]


class UnitsAndPhones(TypedDict):
    """Dictionary mapping filenames to the corresponding predicted units and gold phonemes."""

    units: list[int]
    phones: list[str]


class ABX(TypedDict):
    """Output of ABX evaluation."""

    within: float
    across: float


class DiscoveryEvaluationResult(TypedDict):
    """Output of phoneme discovery evaluation."""

    pnmi: float
    per: float
    f1: float
    r_val: float


class ArgumentsError(ValueError):
    """To raise if a function does not have the correct number of arguments."""

    def __init__(self, n: int = 2) -> None:
        super().__init__(f"Function must have at least {n} positional arguments to compare.")


class ValidateSameKeysError(ValueError):
    """To be raised in the decorator below."""

    def __init__(self) -> None:
        super().__init__("The first two arguments must be dictionaries with the same keys")


def validate_same_keys[R, **P](func: Callable[P, R]) -> Callable[P, R]:
    """Decoractor that checks that the first two arguments of the function are dictionaries with the same keys."""

    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if len(args) < 2:
            raise ArgumentsError
        if not isinstance(args[0], dict) or not isinstance(args[1], dict) or set(args[0]) != set(args[1]):
            raise ValidateSameKeysError
        return func(*args, **kwargs)

    return wrapper
