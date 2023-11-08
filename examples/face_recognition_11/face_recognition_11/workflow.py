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
from kolena.workflow.annotation import ScoredClassificationLabel, ClassificationLabel
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
    normalization_factor: float
    count_genuine_pair: int
    count_imposter_pair: int


@dataclass(frozen=True)
class Inference(BaseInference):
    """Inference type for Face Recognition workflow."""

    similarities: List[Optional[float]]
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
    """Test-sample-level aggregate metrics for each Image Pair."""

    is_match: bool
    is_false_match: bool
    is_false_non_match: bool
    failure_to_enroll: bool
    similarity: Optional[float]


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    """Test-sample-level aggregate metrics for Face Recognition workflow."""

    pair_samples: List[PairSample]
    count_FNM: int
    count_FM: int
    count_TM: int
    count_TNM: int
    similarity_threshold: float
    bbox_IoU: float
    bbox_TP: Optional[List[BoundingBox]]
    bbox_FP: Optional[List[BoundingBox]]
    bbox_FN: Optional[List[BoundingBox]]
    bbox_has_TP: bool
    bbox_has_FP: bool
    bbox_has_FN: bool
    bbox_failure_to_enroll: bool
    keypoint_MSE: Optional[float]
    keypoint_NMSE: Optional[float]
    keypoint_Δ_nose: Optional[float]
    keypoint_Δ_left_eye: Optional[float]
    keypoint_Δ_right_eye: Optional[float]
    keypoint_Δ_left_mouth: Optional[float]
    keypoint_Δ_right_mouth: Optional[float]
    keypoint_norm_Δ_nose: Optional[float]
    keypoint_norm_Δ_left_eye: Optional[float]
    keypoint_norm_Δ_right_eye: Optional[float]
    keypoint_norm_Δ_left_mouth: Optional[float]
    keypoint_norm_Δ_right_mouth: Optional[float]
    keypoint_failure_to_align: bool


@dataclass(frozen=True)
class PerBBoxMetrics(MetricsTestCase):
    """Nested test-case-level aggregate metrics for Face Detection."""

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
    """Nested test-case-level aggregate metrics for Keypoint Detection."""

    Label: str
    Total: int  # number of keypoints
    FTE: int
    MSE: float
    NMSE: float
    AvgΔNose: float
    AvgΔLeftEye: float
    AvgΔRightEye: float
    AvgΔLeftMouth: float
    AvgΔRightMouth: float
    AvgNormΔNose: float
    AvgNormΔLeftEye: float
    AvgNormΔRightEye: float
    AvgNormΔLeftMouth: float
    AvgNormΔRightMouth: float


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    """Test-case-level aggregate metrics for Face Recognition workflow."""

    TotalPairs: int
    nGenuinePairs: int
    nImposterPairs: int
    FM: int
    FMR: float
    FNM: int
    FNMR: float
    ΔFNMR: float
    FTE: int
    FTER: float
    PairFailures: int
    PairFailureRate: float
    PerBBoxMetrics: List[PerBBoxMetrics]
    PerKeypointMetrics: List[PerKeypointMetrics]


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    """Test-suite-level metrics for Face Recognition workflow."""

    Threshold: float
    FM: int
    FNM: int
    FNMR: float
    TotalFTE: int
    TotalBBoxFTE: int
    TotalKeypointFTE: int


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    """
    Configuration for Face Recognition workflow.
    """

    false_match_rate: float
    iou_threshold: float
    nmse_threshold: float

    def display_name(self) -> str:
        return f"False Match Rate: {self.false_match_rate:.1e} | IoU Threshold: {self.iou_threshold} | NMSE threshold: {self.nmse_threshold}"
