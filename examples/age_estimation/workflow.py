from typing import Optional

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Image
from kolena.workflow import Inference as BaseInference


@dataclass(frozen=True)
class TestSample(Image):
    name: str
    race: str
    gender: str
    age: int
    image_width: int
    image_height: int


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    age: float


@dataclass(frozen=True)
class Inference(BaseInference):
    age: Optional[float] = None


_workflow, TestCase, TestSuite, Model = define_workflow("Age Estimation", TestSample, GroundTruth, Inference)
