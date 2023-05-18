from dataclasses import dataclass
from typing import Optional

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as GT
from kolena.workflow import Image
from kolena.workflow import Inference as Inf
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints


@dataclass(frozen=True)
class TestSample(Image):
    bbox: Optional[BoundingBox] = None


@dataclass(frozen=True)
class GroundTruth(GT):
    keypoints: Keypoints

    # In order to compute normalized error, some normalization factor describing
    # the size of the face in the image is required.
    normalization_factor: float


@dataclass(frozen=True)
class Inference(Inf):
    keypoints: Keypoints
    confidence: float


# use these TestCase, TestSuite, and Model definitions to create and run tests
wf, TestCase, TestSuite, Model = define_workflow(
    "Keypoint Detection",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    normalized_mean_error: float

    # If the normalized mean error is above some configured threshold, this test
    # sample is considered an "alignment failure".
    alignment_failure: bool


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    mean_nme: float
    alignment_failure_rate: float


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    variance_mean_nme: float
    variance_alignment_failure_rate: float


@dataclass(frozen=True)
class NmeThreshold(EvaluatorConfiguration):
    # threshold for NME (Normalized Mean Error) above which an image is considered an "alignment failure"
    threshold: float

    def display_name(self):
        return f"nme-threshold-{self.threshold}"
