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
from kolena.workflow import Text
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.test_sample import BaseVideo


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for the pre-built Classification workflow."""

    classification: ClassificationLabel
    """The classfication label associated with an image."""


@dataclass(frozen=True)
class Inference(BaseInference):
    """Inference type for the pre-built Classification workflow."""

    inferences: List[ScoredClassificationLabel]
    """
    The model predictions for an image. For `N`-class problems, `inferences` is expected to contain `N` entries, one for
    each class and its associated confidence score.
    """


@dataclass(frozen=True)
class ClassificationTestSample(Text):
    """Test sample type for the pre-built Classification workflow."""

    metadata: Metadata = dataclasses.field(default_factory=dict)
    """The metadata associated with the test sample."""


_workflow, _TestCase, _TestSuite, _Model = define_workflow(
    "Classification",
    ClassificationTestSample,
    GroundTruth,
    Inference,
)
"""Example"""


TestCase = _TestCase
"""[`TestCase`][kolena.workflow.TestCase] definition bound to the pre-built Classification workflow."""


TestSuite = _TestSuite
"""[`TestSuite`][kolena.workflow.TestSuite] definition bound to the pre-built Classification workflow."""


Model = _Model
"""[`Model`][kolena.workflow.Model] definition bound to the pre-built Classification workflow."""


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    """TestSample-level metrics for the pre-built Classification workflow."""

    classification_label: Optional[str]
    """
    The model's classification label of greatest confidence score. Empty when no inference has a sufficient confidence
    score based on the [`ThresholdConfiguration`][kolena._experimental.classification.workflow.ThresholdConfiguration].
    """

    classification_score: Optional[float]
    """
    The model's confidence score for the `classification_label`. Empty when no inference has a sufficient confidence
    score based on the [`ThresholdConfiguration`][kolena._experimental.classification.workflow.ThresholdConfiguration].
    """

    margin: Optional[float]
    """
    The difference in confidence scores between the `classification_score` and the inference with the second-highest
    confidence score. Empty when no prediction with sufficient confidence score is provided.
    """

    is_correct: bool
    """An indication of the `classification_label` matching the associated ground truth label."""


@dataclass(frozen=True)
class ClassMetricsPerTestCase(MetricsTestCase):
    label: str
    n_correct: int
    n_incorrect: int
    accuracy: float
    Precision: float
    Recall: float
    F1: float
    FPR: float


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    PerClass: List[ClassMetricsPerTestCase]
    n_labels: int
    n_correct: int
    n_incorrect: int
    macro_accuracy: float
    macro_Precision: float
    macro_Recall: float
    macro_F1: float
    macro_FPR: float


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    n_images: int
    n_invalid: int
    n_correct: int
    overall_accuracy: float


class ThresholdStrategy(str, Enum):
    F1_OPTIMAL = "F1_OPTIMAL"
    FIXED_05 = "FIXED_05"

    def display_name(self) -> str:
        if self is ThresholdStrategy.FIXED_05:
            return "Fixed(0.5)"
        if self is ThresholdStrategy.F1_OPTIMAL:
            return "F1-Optimal"
        raise RuntimeError(f"unrecognized threshold strategy: {self}")


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    threshold_strategy: ThresholdStrategy
    min_confidence_score: float

    def display_name(self) -> str:
        return f"Threshold: {self.threshold_strategy.display_name()}, " f"confidence ≥ {self.min_confidence_score}"


class WORKFLOW_TYPES(str, Enum):
    IMAGE = "Image"
    VIDEO = "Video"
    TEXT = "Text"


@dataclass(frozen=True)
class ImageTestSample(Image):
    metadata: Metadata = dataclasses.field(default_factory=dict)


ImageWorkflow, ImageTestCase, ImageTestSuite, ImageModel = define_workflow(
    "Image Classification",
    ImageTestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class VideoTestSample(BaseVideo):
    metadata: Metadata = dataclasses.field(default_factory=dict)


VideoWorkflow, VideoTestCase, VideoTestSuite, VideoModel = define_workflow(
    "Video Classification",
    VideoTestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class TextTestSample(Text):
    metadata: Metadata = dataclasses.field(default_factory=dict)


TextWorkflow, TextTestCase, TextTestSuite, TextModel = define_workflow(
    "Text Classification",
    TextTestSample,
    GroundTruth,
    Inference,
)
