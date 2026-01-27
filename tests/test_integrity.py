from functools import partial
from pathlib import Path

import pytest
from fastabx.dataset import read_labels

from discophon.benchmark import validate_dataset_structure
from discophon.data import read_gold_annotations_as_dataframe
from discophon.languages import all_languages


@pytest.mark.requires_dataset
def test_validate_dataset_structure(dataset_path: Path) -> None:
    validate_dataset_structure(dataset_path)


@pytest.mark.requires_dataset
def test_inventory(dataset_path: Path) -> None:
    read_item = partial(read_labels, file_col="#file", onset_col="onset", offset_col="offset")
    for language in all_languages():
        for split in ("dev", "test"):
            alignment_path = dataset_path / "alignment" / f"alignment-{language.iso_639_3}-{split}.txt"
            alignments = read_gold_annotations_as_dataframe(alignment_path)
            assert sorted(alignments["#phone"].unique()) == ["SIL", *language.phonology]
            for kind in ("triphone", "phoneme"):
                item = read_item(dataset_path / "item" / f"{kind}-{language.iso_639_3}-{split}.item")
                assert sorted(item["#phone"].unique()) == language.phonology
                assert sorted(item["next-phone"].unique()) == language.phonology
                assert sorted(item["prev-phone"].unique()) == language.phonology
