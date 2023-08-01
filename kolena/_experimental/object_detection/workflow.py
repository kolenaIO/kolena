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
    """The [`Image`][kolena.workflow.Image] sample type for the pre-built 2D Object Detection workflow."""

    metadata: Metadata = dataclasses.field(default_factory=dict)
    """The optional [`Metadata`][kolena.workflow.Metadata] dictionary."""


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for the pre-built 2D Object Detection workflow."""

    bboxes: List[LabeledBoundingBox]
    """
    The ground truth [`LabeledBoundingBox`][kolena.workflow.annotation.LabeledBoundingBox]es associated with an image.
    """

    ignored_bboxes: List[LabeledBoundingBox] = dataclasses.field(default_factory=list)
    """
    The ground truth [`LabeledBoundingBox`][kolena.workflow.annotation.LabeledBoundingBox]es to be ignored in evaluation
    associated with an image.
    """

    labels: List[str] = dataclasses.field(default_factory=list)
    n_bboxes: int = dataclasses.field(default_factory=lambda: 0)

    def __post_init__(self):
        object.__setattr__(self, "labels", sorted({box.label for box in self.bboxes}))
        object.__setattr__(self, "n_bboxes", len(self.bboxes))


@dataclass(frozen=True)
class Inference(BaseInference):
    """Inference type for the pre-built 2D Object Detection workflow."""

    bboxes: List[ScoredLabeledBoundingBox]
    """
    The inference [`ScoredLabeledBoundingBox`][kolena.workflow.annotation.ScoredLabeledBoundingBox]es associated with
    an image.
    """
    ignored: bool = False
    """
    Whether the image (and its associated inference `bboxes`) should be ignored in evaluating the results of the model.
    """


_, TestCase, TestSuite, Model = define_workflow(
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
    ignored: bool

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
    nIgnored: int
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
    ignored: bool

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
    nIgnored: int
    macro_Precision: float
    macro_Recall: float
    macro_F1: float
    mean_AP: float


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    n_images: int
    mean_AP: float


class ThresholdStrategy(str, Enum):
    """
    Threshold strategy enumerations used in
    [`ThresholdConfiguration`][kolena._experimental.object_detection.ThresholdConfiguration].
    """

    F1_OPTIMAL = "F1_OPTIMAL"
    """Confidence threshold that yields the most optimal F1-score."""
    FIXED_03 = "FIXED_03"
    """Confidence threshold fixed at 0.3."""
    FIXED_05 = "FIXED_05"
    """Confidence threshold fixed at 0.5."""
    FIXED_075 = "FIXED_075"
    """Confidence threshold fixed at 0.75."""

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
    """
    Confidence and [IoU ↗](../../metrics/iou.md) threshold configuration for the pre-built 2D Object Detection workflow.
    Specify a confidence and IoU threshold to apply to all classes.
    """

    threshold_strategy: ThresholdStrategy
    """The confidence threshold strategy."""

    iou_threshold: float
    """The [IoU ↗](../../metrics/iou.md) threshold."""

    with_class_level_metrics: bool
    """
    The flag that enables multiclass evaluation. If it's set to `False` on a multiclass `TestSuite`, it will only
    perform localization evaluation by treating it as a binary class problem.
    """

    min_confidence_score: float = 0.0
    """
    The minimum confidence score to consider for the evaluation. This is usually set to reduce noise by excluding
    inferences with low confidence score.
    """

    def display_name(self) -> str:
        return (
            f"Threshold: {self.threshold_strategy.display_name()}"
            f"{' by class' if self.with_class_level_metrics else ''}, "
            f"IoU: {self.iou_threshold}, confidence ≥ {self.min_confidence_score}"
        )
