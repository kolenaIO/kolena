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
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.annotation import ScoredLabeledBoundingBox


@dataclass(frozen=True)
class TestSample(Image):
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    bboxes: List[LabeledBoundingBox]
    ignored_bboxes: List[LabeledBoundingBox] = dataclasses.field(default_factory=list)
    labels: List[str] = dataclasses.field(default_factory=list)
    n_bboxes: int = dataclasses.field(default_factory=lambda: 0)

    def __post_init__(self):
        object.__setattr__(self, "labels", sorted({box.label for box in self.bboxes}))
        object.__setattr__(self, "n_bboxes", len(self.bboxes))


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
class TestSampleMetricsSingleClass(MetricsTestSample):
    TP: List[ScoredLabeledBoundingBox]
    FP: List[ScoredLabeledBoundingBox]
    FN: List[LabeledBoundingBox]

    count_TP: int
    count_FP: int
    count_FN: int

    has_TP: bool
    has_FP: bool
    has_FN: bool

    max_confidence_above_t: Optional[float]
    min_confidence_above_t: Optional[float]
    thresholds: float


@dataclass(frozen=True)
class TestCaseMetricsSingleClass(MetricsTestCase):
    Objects: int
    Inferences: int
    TP: int
    FN: int
    FP: int
    Precision: float
    Recall: float
    F1: float
    AP: float


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    TP_labels: List[str]
    TP: List[ScoredLabeledBoundingBox]
    FP_labels: List[str]
    FP: List[ScoredLabeledBoundingBox]
    FN_labels: List[str]
    FN: List[LabeledBoundingBox]
    Confused_labels: List[str]
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
    thresholds: List[ScoredClassificationLabel]
    inference_labels: List[str]


@dataclass(frozen=True)
class ClassMetricsPerTestCase(MetricsTestCase):
    Class: str
    nImages: int
    Threshold: float
    Objects: int
    Inferences: int
    TP: int
    FN: int
    FP: int
    Precision: float
    Recall: float
    F1: float
    AP: float


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    PerClass: List[ClassMetricsPerTestCase]
    Objects: int
    Inferences: int
    TP: int
    FN: int
    FP: int
    macro_Precision: float
    macro_Recall: float
    macro_F1: float
    mean_AP: float


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    n_images: int
    mean_AP: float


class ThresholdStrategy(str, Enum):
    F1_OPTIMAL = "F1_OPTIMAL"
    FIXED_03 = "FIXED_03"
    FIXED_05 = "FIXED_05"
    FIXED_075 = "FIXED_075"

    def display_name(self) -> str:
        if self is ThresholdStrategy.FIXED_03:
            return "Fixed(0.3)"
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
    with_class_level_metrics: bool
    min_confidence_score: float = 0.0

    def display_name(self) -> str:
        return (
            f"Threshold: {self.threshold_strategy.display_name()}"
            f"{' by class' if self.with_class_level_metrics else ''}, "
            f"IoU: {self.iou_threshold}, confidence ≥ {self.min_confidence_score}"
        )
