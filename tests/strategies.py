"""Shared Hypothesis strategies and helpers for the test suite."""

from hypothesis import strategies as st

# Small alphabets keep the search space meaningful while exercising real behaviour.
PHONE_ALPHABET = ("a", "b", "c", "d", "SIL")


def phone_sequences(min_size: int = 0, max_size: int = 30) -> st.SearchStrategy[list[str]]:
    """Sequences of phone-like tokens."""
    return st.lists(st.sampled_from(PHONE_ALPHABET), min_size=min_size, max_size=max_size)


def int_sequences(max_value: int = 5, min_size: int = 0, max_size: int = 30) -> st.SearchStrategy[list[int]]:
    """Sequences of small non-negative integers (unit-like tokens)."""
    return st.lists(st.integers(min_value=0, max_value=max_value), min_size=min_size, max_size=max_size)


def reference_edit_distance(a: list, b: list) -> int:
    """Plain dynamic-programming Levenshtein distance, used as an oracle in tests."""
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + cost))
        previous = current
    return previous[-1]
