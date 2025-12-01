from typing import Literal, get_args

from discophon.core import DevLanguage, Language, TestLanguage
from pydantic import BaseModel, FilePath, PositiveInt, RootModel, ValidationInfo, model_validator


def validate_split(split: Literal["full", "dev", "test"], keys: set[str]) -> None:
    match split:
        case "full":
            expected = set(get_args(DevLanguage.__value__)) | set(get_args(TestLanguage.__value__))
        case "dev":
            expected = set(get_args(DevLanguage.__value__))
        case "test":
            expected = set(get_args(TestLanguage.__value__))
        case _:
            raise ValueError(f"Invalid split: {split}")
    if keys != expected:
        raise ValueError(f"Invalid keys for {split=}: expected {expected}, found {keys}")


class RootModelWithSplitVerification(RootModel):
    @model_validator(mode="after")
    def check_split(self, info: ValidationInfo) -> None:
        if not isinstance(info.context, dict) or "split" not in info.context:
            return
        validate_split(info.context["split"], set(self.root.keys()))


class Submission(BaseModel):
    units: FilePath
    n_units: PositiveInt
    step_units: PositiveInt


class Submissions(RootModelWithSplitVerification):
    root: dict[Language, Submission]


class Annotation(BaseModel):
    phones: FilePath
    n_phones: PositiveInt
    step_phones: PositiveInt


class Annotations(RootModelWithSplitVerification):
    root: dict[Language, Annotation]


class PhonemeDiscoveryScoreEntity(BaseModel):
    score: float


class PhonemeDiscoveryScores(BaseModel):
    pnmi: dict[Language, PhonemeDiscoveryScoreEntity]
    per: dict[Language, PhonemeDiscoveryScoreEntity]
    f1: dict[Language, PhonemeDiscoveryScoreEntity]
    r_val: dict[Language, PhonemeDiscoveryScoreEntity]
