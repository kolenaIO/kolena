# build out the workflow for the FR problem which is already supported in kolena.fr as a pre-built workflow.
# Copyright 2021-2023 Kolena Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import dataclasses
from typing import Optional
from typing import List

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import DataObject
from kolena.workflow import Image
from kolena.workflow import Composite
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.asset import ImageAsset


@dataclass(frozen=True)
class TestSample(Image):
    """Test sample type for Face Recognition 1:1 workflow."""

    pairs: List[ImageAsset]
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for Face Recognition 1:1 workflow."""

    matches: List[bool]
    bbox: BoundingBox
    keypoints: Keypoints


@dataclass(frozen=True)
class Inference(BaseInference):
    """Inference type for Face Recognition 1:1 workflow."""

    similarities: List[Optional[float]]
    """
    The similarity score computed between the two embeddings in this image pair. Should be left empty when either
    image in the pair is a failure to enroll.
    """

    bbox: Optional[BoundingBox]
    keypoints: Optional[Keypoints]


workflow, TestCase, TestSuite, Model = define_workflow(
    "Face Recognition 1:1",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class PairSample(ImageAsset):
    is_match: bool
    is_false_match: bool
    is_false_non_match: bool
    failure_to_enroll: bool
    similarity: Optional[float]


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    pair_samples: List[PairSample]
    bbox_iou: ScoredClassificationLabel
    bbox_tp: bool
    bbox_fp: bool
    bbox_fn: bool
    keypoint_mse: ScoredClassificationLabel
    keypoint_Δ_nose: float
    keypoint_Δ_left_eye: float
    keypoint_Δ_right_eye: float
    keypoint_Δ_left_mouth: float
    keypoint_Δ_right_mouth: float


@dataclass(frozen=True)
class PerBBoxMetrics(MetricsTestCase):
    Label: str
    Total: int
    FTE: int
    AvgIoU: float
    Precision: float
    Recall: float
    F1: float
    TP: int
    FP: int
    FN: int


@dataclass(frozen=True)
class PerKeypointMetrics(MetricsTestCase):
    Label: str
    Total: int  # number of keypoints
    FTE: int
    MSE: float
    AvgΔNose: float
    AvgΔLeftEye: float
    AvgΔRightEye: float
    AvgΔLeftMouth: float
    AvgΔRightMouth: float


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    """Test-case-level aggregate metrics for Face Recognition 1:1 workflow."""

    nImages: int
    """Total number of unique images within this test case."""

    nGenuinePairs: int
    """Total number of genuine pairs within this test case."""

    nImposterPairs: int
    """Total number of imposter pairs within this test case."""

    FM: int
    """Total number of false matches within this test case."""

    FMR: float
    """
    The percentage of imposter pairs that are incorrectly classified as genuine pairs
    (i.e. similarity is above threshold) within this test case.
    """

    FNM: int
    """Total number of false non-matches within this test case."""

    FNMR: float
    """
    The percentage of genuine pairs that are incorrectly classified as imposter pairs
    (i.e. similarity is below threshold) within this test case.
    """

    ΔFNMR: float
    """
    The percentage difference of each test case compared to the baseline.
    """

    FTE: int
    """Total number of failure to enroll (FTE) across images within this test case."""

    FTER: float
    """The percentage of FTE across images that exist within the test case."""

    PairFailures: int
    """Total number of failure to enroll (FTE) across test samples within this test case."""

    PairFailureRate: float
    """The percentage of FTE across test samples within the test case."""

    PerBBoxMetrics: List[PerBBoxMetrics]

    PerKeypointMetrics: List[PerKeypointMetrics]


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    """Test-suite-level metrics for Face Recognition 1:1 workflow."""

    Threshold: float
    """The threshold value of the baseline test case given a specific FMR."""

    FM: int
    """The total number of false matches in a """

    FNM: int
    """The threshold value of the baseline test case given a specific FMR."""

    FNMR: float
    """The threshold value of the baseline test case given a specific FMR."""


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    """
    Configuration for Face Recognition 1:1 workflow.
    """

    false_match_rate: Optional[float] = None
    """
    Specify a minimum FMR to apply for predictions.
    """

    iou_threshold: Optional[float] = None

    def display_name(self) -> str:
        return f"False Match Rate: {self.false_match_rate:.1e} | IoU Threshold: {self.iou_threshold}"
