from collections.abc import Callable
from functools import wraps
from typing import TypedDict

type Units = dict[str, list[int]]
type Phones = dict[str, list[str]]


class UnitsAndPhones(TypedDict):
    units: list[int]
    phones: list[str]


class ABX(TypedDict):
    within: float
    across: float


class DiscoveryEvaluationResult(TypedDict):
    pnmi: float
    per: float
    f1: float
    r_val: float


class ArgumentsError(ValueError):
    def __init__(self) -> None:
        super().__init__("Function must have at least two positional arguments to compare.")


class SameKeysError(ValueError):
    def __init__(self) -> None:
        super().__init__("The first two arguments must be dictionnaries with the same keys")


def validate_same_keys[R, **P](func: Callable[P, R]) -> Callable[P, R]:
    @wraps(func)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        if len(args) < 2:
            raise ArgumentsError
        if not isinstance(args[0], dict) or not isinstance(args[1], dict) or set(args[0]) != set(args[1]):
            raise SameKeysError
        return func(*args, **kwargs)

    return wrapper
