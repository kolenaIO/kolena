from dataclasses import field
from typing import Dict

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth
from kolena.workflow import Image
from kolena.workflow import Inference
from kolena.workflow.annotation import BoundingBox

DUMMY_WORKFLOW_NAME = "Dummy Workflow 🤖"


@dataclass(frozen=True, order=True)
class DummyTestSample(Image):
    value: int
    bbox: BoundingBox
    metadata: Dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, order=True)
class DummyGroundTruth(GroundTruth):
    label: str
    value: int


@dataclass(frozen=True, order=True)
class DummyInference(Inference):
    score: float


DUMMY_WORKFLOW, TestCase, TestSuite, Model = define_workflow(
    name=DUMMY_WORKFLOW_NAME,
    test_sample_type=DummyTestSample,
    ground_truth_type=DummyGroundTruth,
    inference_type=DummyInference,
)
