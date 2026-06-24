from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--dataset",
        action="store",
        default=None,
        type=Path,
        help="Path to the Phoneme Discovery dataset for optional tests",
    )
    parser.addoption(
        "--tipa_output",
        action="store",
        default=None,
        type=Path,
        help="Output path for the TIPA mapping verification plot. Required to run the TIPA mapping test.",
    )
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="Run slow tests (e.g. the model batching-equivalence tests)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--dataset") is None:
        skip_marker = pytest.mark.skip(reason="Need --dataset option to run")
        for item in items:
            if "requires_dataset" in item.keywords:
                item.add_marker(skip_marker)
    if config.getoption("--tipa_output") is None:
        skip_marker = pytest.mark.skip(reason="Need --tipa_output option to run")
        for item in items:
            if "requires_tipa_output" in item.keywords:
                item.add_marker(skip_marker)
    if not config.getoption("--run-slow"):
        skip_marker = pytest.mark.skip(reason="Need --run-slow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_marker)


@pytest.fixture
def dataset_path(request: pytest.FixtureRequest) -> Path:
    """Fixture that provides the dataset path to tests."""
    return request.config.getoption("--dataset")


@pytest.fixture
def tipa_output(request: pytest.FixtureRequest) -> Path:
    """Fixture that provides the output path for the TIPA mapping plot."""
    return request.config.getoption("--tipa_output")
