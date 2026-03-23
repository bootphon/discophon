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


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    dataset_path: str | None = config.getoption("--dataset")
    if dataset_path is None:
        skip_marker = pytest.mark.skip(reason="Need --dataset option to run")
        for item in items:
            if "requires_dataset" in item.keywords:
                item.add_marker(skip_marker)


@pytest.fixture
def dataset_path(request: pytest.FixtureRequest) -> Path:
    """Fixture that provides the dataset path to tests."""
    return request.config.getoption("--dataset")
