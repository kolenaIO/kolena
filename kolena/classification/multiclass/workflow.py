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
from typing import List
from typing import Optional

from pydantic import Field
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
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import ScoredClassificationLabel


@dataclass(frozen=True)
class TestSample(Image):
    """
    The test sample type for the pre-built Multiclass Classification workflow. Extends [`Image`][kolena.workflow.Image].
    """

    metadata: Metadata = Field(default_factory=dict)
    """Free-form metadata to associate with a test sample."""


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for the pre-built Multiclass Classification workflow."""

    classification: ClassificationLabel
    """The class label for an image."""


@dataclass(frozen=True)
class Inference(BaseInference):
    inferences: List[ScoredClassificationLabel]
    """
    Model predictions for an image. For `N`-class problems, `inferences` is expected to contain `N` entries, one for
    each class and its associated confidence score.
    """


_workflow, _TestCase, _TestSuite, _Model = define_workflow(
    "Multiclass Classification",
    TestSample,
    GroundTruth,
    Inference,
)
"""Example"""

TestCase = _TestCase
"""[`TestCase`][kolena.workflow.TestCase] definition bound to the pre-built Multiclass Classification workflow."""

TestSuite = _TestSuite
"""[`TestSuite`][kolena.workflow.TestSuite] definition bound to the pre-built Multiclass Classification workflow."""

Model = _Model
"""[`Model`][kolena.workflow.Model] definition bound to the pre-built Multiclass Classification workflow."""


@dataclass(frozen=True)
class PerImageMetrics(MetricsTestSample):
    """Image-level metrics for the pre-built Multiclass Classification workflow."""

    classification: Optional[ScoredClassificationLabel]
    """
    The model classification for this image. Empty when no inference with sufficient confidence score is provided
    when using a non-argmax [`ThresholdConfiguration`][kolena.classification.multiclass.ThresholdConfiguration].
    """

    margin: Optional[float]
    """
    The margin in confidence score between the selected `classification` and the inference with the second-highest
    confidence score. Empty when no prediction with sufficient confidence score is provided.
    """

    is_correct: bool
    """Status of the `classification`, i.e. `classification.label` matches the associated ground truth label."""


@dataclass(frozen=True)
class PerClassMetrics:
    """Class-level aggregate metrics for a single class within a test case."""

    label: str
    """Label associated with this class."""

    Precision: float
    """Precision score for this class within this test case."""

    Recall: float
    """Recall score for this class within this test case."""

    F1: float
    """F1 score score for this class within this test case."""

    FPR: float
    """False positive rate (FPR) for this class within this test case."""


@dataclass(frozen=True)
class AggregateMetrics(MetricsTestCase):
    """Test-case-level aggregate metrics for the pre-built Multiclass Classification workflow."""

    n_correct: int
    """Total number of correct predictions within this test case."""

    n_incorrect: int
    """Total number of incorrect predictions within this test case."""

    accuracy: float
    """Accuracy for predictions within this test case."""

    macro_precision: float
    """Macro-averaged precision score for all classes within this test case."""

    macro_recall: float
    """Macro-averaged recall score for all classes within this test case."""

    macro_f1: float
    """Macro-averaged F1 score for all classes within this test case."""

    macro_tpr: float
    """Macro-averaged true positive rate (TPR) score for all classes within this test case."""

    macro_fpr: float
    """Macro-averaged false positive rate (FPR) score for all classes within this test case."""


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    """Test-suite-level metrics for the pre-built Multiclass Classification workflow."""

    n_images: int
    """Total number of images tested within this test suite."""

    n_correct: int
    """Total number of correct predictions within this test suite."""

    mean_test_case_accuracy: float
    """Macro-averaged accuracy across all test cases within this test suite."""


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    """
    Confidence threshold configuration for the pre-built Multiclass Classification workflow.

    Specify a fixed `threshold` to apply to all classes, or leave unspecified to select the class with the highest
    confidence.
    """

    threshold: Optional[float] = None
    """
    Confidence score threshold to apply for predictions. When empty, the prediction with highest confidence is used.
    When non-empty, the highest confidence prediction is considered only if it is above this threshold.
    """

    def display_name(self) -> str:
        if self.threshold:
            return f"Confidence Above Threshold (threshold={self.threshold})"
        return "Max Confidence"
