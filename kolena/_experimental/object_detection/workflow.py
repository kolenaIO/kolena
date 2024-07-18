# Copyright 2021-2024 Kolena Inc.
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
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from kolena._utils.pydantic_v1.dataclasses import dataclass
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
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import ScoredLabel
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledPolygon


@dataclass(frozen=True)
class TestSample(Image):
    """The [`Image`][kolena.workflow.Image] sample type for the pre-built 2D Object Detection workflow."""

    metadata: Metadata = dataclasses.field(default_factory=dict)
    """The optional [`Metadata`][kolena.workflow.Metadata] dictionary."""


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for the pre-built 2D Object Detection workflow."""

    bboxes: List[Union[LabeledBoundingBox, LabeledPolygon]]
    """
    The ground truth [`LabeledBoundingBox`][kolena.workflow.annotation.LabeledBoundingBox]es or
    [`LabeledPolygon`][kolena.workflow.annotation.LabeledPolygon]s associated with an image.
    """

    ignored_bboxes: List[Union[LabeledBoundingBox, LabeledPolygon]] = dataclasses.field(default_factory=list)
    """
    The ground truth [`LabeledBoundingBox`][kolena.workflow.annotation.LabeledBoundingBox]es or
    [`LabeledPolygon`][kolena.workflow.annotation.LabeledPolygon]s to be ignored in evaluation associated with an
    image.
    """

    n_bboxes: int = dataclasses.field(default_factory=lambda: 0)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_bboxes", len(self.bboxes))


@dataclass(frozen=True)
class Inference(BaseInference):
    """Inference type for the pre-built 2D Object Detection workflow."""

    bboxes: List[Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon]]
    """
    The inference [`ScoredLabeledBoundingBox`][kolena.workflow.annotation.ScoredLabeledBoundingBox]es or
    [`ScoredLabeledPolygon`][kolena.workflow.annotation.ScoredLabeledPolygon]s associated with an image.
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
    TP: List[Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon]]
    FP: List[Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon]]
    FN: List[Union[LabeledBoundingBox, LabeledPolygon]]

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
    TP: List[Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon]]
    FP: List[Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon]]
    FN: List[Union[LabeledBoundingBox, LabeledPolygon]]
    Confused: List[Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon]]

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
    thresholds: List[ScoredLabel]


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
    micro_Precision: float
    micro_Recall: float
    micro_F1: float


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    n_images: int
    mean_AP: float
    threshold: Optional[float] = None


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    """
    Confidence and [IoU ↗](../../metrics/iou.md) threshold configuration for the pre-built 2D Object Detection workflow.
    Specify a confidence and IoU threshold to apply to all classes.
    """

    threshold_strategy: Union[Literal["F1-Optimal"], float] = "F1-Optimal"
    """The confidence threshold strategy. It can either be a fixed confidence threshold such as `0.3` or `0.75`, or
    the F1-optimal threshold by default."""

    iou_threshold: float = 0.5
    """The [IoU ↗](../../metrics/iou.md) threshold, defaulting to `0.5`."""

    min_confidence_score: float = 0.0
    """
    The minimum confidence score to consider for the evaluation. This is usually set to reduce noise by excluding
    inferences with low confidence score.
    """

    multiclass: Optional[bool] = None
    """Optional flag to choose whether the SingleClass or Multiclass evaluator is run.
    If unset, the evaluator is determined based on the number of distinct labels in test data."""

    def display_name(self) -> str:
        evaluator_name = ""
        if self.multiclass is not None:
            evaluator_name = "Multiclass, " if self.multiclass else "SingleClass, "
        return (
            f"{evaluator_name}"
            f"Confidence Threshold: {self.threshold_strategy}, "
            f"IoU: {self.iou_threshold}, "
            f"min confidence ≥ {self.min_confidence_score}"
        )
