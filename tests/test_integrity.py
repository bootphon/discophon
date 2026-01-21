from pathlib import Path

import pytest

from discophon.benchmark import validate_dataset_structure


@pytest.mark.requires_dataset
def test_validate_dataset_structure(dataset_path: Path) -> None:
    validate_dataset_structure(dataset_path)
