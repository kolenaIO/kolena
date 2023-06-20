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
from enum import Enum
from typing import List
from typing import Optional

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Image
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox


@dataclass(frozen=True)
class TestSample(Image):
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    bboxes: List[LabeledBoundingBox]
    ignored_bboxes: List[LabeledBoundingBox]


@dataclass(frozen=True)
class Inference(BaseInference):
    bboxes: List[ScoredLabeledBoundingBox]


_workflow, TestCase, TestSuite, Model = define_workflow(
    "Object Detection",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    TP: List[ScoredLabeledBoundingBox]
    FP: List[ScoredLabeledBoundingBox]
    FN: List[LabeledBoundingBox]
    Confused: List[ScoredLabeledBoundingBox]

    count_TP: int
    count_FP: int
    count_FN: int
    count_Confused: int

    has_TP: bool
    has_FP: bool
    has_FN: bool
    has_Confused: bool

    max_confidence_above_t: Optional[float]
    min_confidence_above_t: Optional[float]


@dataclass(frozen=True)
class ClassMetricsPerTestCase(MetricsTestCase):
    Class: str
    TestSamples: int
    Objects: int
    Inferences: int
    TP: int
    FN: int
    FP: int
    TPR: float
    FPR: float
    Precision: float
    Recall: float
    F1: float
    AP: float


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    TestSamples: int
    PerClass: List[ClassMetricsPerTestCase]
    Objects: int
    Inferences: int
    TP: int
    FN: int
    FP: int
    TPR: float
    FPR: float
    Precision: float
    Recall: float
    F1: float
    AP: float


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    n_images: int
    mean_AP: float
    variance_AP: float


class ThresholdStrategy(str, Enum):
    F1_OPTIMAL = "F1_OPTIMAL"
    FIXED_05 = "FIXED_05"
    FIXED_075 = "FIXED_075"

    def display_name(self) -> str:
        if self is ThresholdStrategy.FIXED_05:
            return "Fixed(0.5)"
        if self is ThresholdStrategy.FIXED_075:
            return "Fixed(0.75)"
        if self is ThresholdStrategy.F1_OPTIMAL:
            return "F1-Optimal"
        raise RuntimeError(f"unrecognized threshold strategy: {self}")


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    threshold_strategy: ThresholdStrategy
    iou_threshold: float
    min_confidence_score: float

    def display_name(self) -> str:
        return (
            f"Threshold: {self.threshold_strategy.display_name()}, "
            f"IOU: {self.iou_threshold}, confidence ≥ {self.min_confidence_score}"
        )
