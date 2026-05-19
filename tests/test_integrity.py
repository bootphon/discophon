from functools import partial
from pathlib import Path

import pytest
from fastabx.dataset import read_labels

from discophon.benchmark import validate_dataset_structure
from discophon.data import alignment_filename, item_filename, read_gold_annotations_as_dataframe
from discophon.languages import all_languages


@pytest.mark.requires_dataset
def test_validate_dataset_structure(dataset_path: Path) -> None:
    validate_dataset_structure(dataset_path)


@pytest.mark.requires_dataset
def test_inventory(dataset_path: Path) -> None:
    read_item = partial(read_labels, file_col="#file", onset_col="onset", offset_col="offset")
    for language in all_languages():
        for split in ("dev", "test"):
            alignments = read_gold_annotations_as_dataframe(
                dataset_path / "alignment" / alignment_filename(language, split)
            )
            assert sorted(alignments["#phone"].unique()) == ["SIL", *language.phonemes]
            for kind in ("triphone", "phoneme"):
                item = read_item(dataset_path / "item" / item_filename(language, split, kind=kind))
                assert sorted(item["#phone"].unique()) == language.phonemes
                assert sorted(item["next-phone"].unique()) == language.phonemes
                assert sorted(item["prev-phone"].unique()) == language.phonemes
