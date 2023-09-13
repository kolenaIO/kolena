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
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import ScoredClassificationLabel


@dataclass(frozen=True)
class TestSample(Image):
    """Test sample type for Classification workflow."""

    metadata: Metadata = dataclasses.field(default_factory=dict)
    """The metadata associated with the test sample."""


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for Classification workflow."""

    classification: ClassificationLabel
    """
    The classfication label associated with an image.

    For binary classification, the negative class should also be labeled. (e.g., `dog` vs. `not dog` or `dog` vs. `cat`)
    """


@dataclass(frozen=True)
class Inference(BaseInference):
    """Inference type for the multiclass classification workflow."""

    inferences: List[ScoredClassificationLabel]
    """
    The model inferences for an image. For `N`-class problems, `label` is expected to contain `N` entries, one for
    each class and its associated confidence score.

    For binary classification, only positive class's confidence score is required. `ThresholdConfiguration` will be
    applied to the positive class only. If inferences for both positive and negative class are provided, then it will
    be treated as multiclass classification.
    """

    ignored: bool = False
    """
    Whether the image should be ignored in evaluating the results of the model.
    """


workflow, TestCase, TestSuite, Model = define_workflow(
    "Image Classification",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class TestSampleMetricsSingleClass(MetricsTestSample):
    """
    Image-level metrics for Binary Classification workflow.

    It is evaluated as a binary classification workflow when there is only
    one label's prediction provided as part of `Inference`, and this label is considered
    to be positive.
    """

    is_correct: bool
    """An indication of the `classification_label` matching the associated ground truth label."""

    is_TP: bool
    """
    An indication of the `classification_label` correctly matching the associated positive ground
    truth label.
    """

    is_FP: bool
    """
    An indication of the `classification_label` incorrectly matching the associated negative ground
    truth label.
    """

    is_FN: bool
    """
    An indication of the `classification_label` incorrectly matching the associated positive ground
    truth label.
    """

    is_TN: bool
    """
    An indication of the `classification_label` correctly matching the associated negative ground
    truth label.
    """

    classification: ScoredClassificationLabel
    """
    The model inference for this image.
    """

    threshold: float
    """
    The threshold used in evaluation — specified by the `ThresholdConfiguration`. It is commonly used in a binary
    classification problem.
    """

    missing_inference: bool = False

    def __post_init__(self):
        object.__setattr__(self, "missing_inference", self.classification is None)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    """Image-level metrics for Multiclass Classification workflow."""

    is_correct: bool
    """An indication of the `classification_label` matching the associated ground truth label."""

    classification: Optional[ScoredClassificationLabel]
    """
    The model classification for this image. Empty when no inferences have a sufficient confidence score determined
    by the `ThresholdConfiguration`.
    """

    margin: Optional[float] = None
    """
    The difference in confidence scores between the `classification_score` and the inference with the second-highest
    confidence score. Empty when no prediction with sufficient confidence score is provided or when it's a binary
    classification.
    """

    threshold: Optional[float] = None
    """
    The threshold used in evaluation — specified by the `ThresholdConfiguration`. It is commonly used in a binary
    classification problem.
    """

    missing_inference: bool = False

    def __post_init__(self):
        object.__setattr__(self, "missing_inference", self.classification is None)


@dataclass(frozen=True)
class TestCaseMetricsSingleClass(MetricsTestCase):
    """
    Test-case-level aggregate metrics for Binary Classification workflow.

    It is evaluated as a binary classification workflow when there is only
    one label's prediction provided as part of `Inference`, and this label is considered
    to be positive.
    """

    TP: int
    """Total number of true positives within this test case."""

    FP: int
    """Total number of false positives within this test case."""

    FN: int
    """Total number of false negatives within this test case."""

    TN: int
    """Total number of true negatives within this test case."""

    Accuracy: float
    """Accuracy score for predictions within this test case."""

    Precision: float
    """Precision score for predictions within this test case."""

    Recall: float
    """Recall score for predictions within this test case. Equivalent to **True Positive Rate** (TPR)."""

    F1: float
    """F1 score for predictions within this test case."""

    FPR: float
    """FPR score for predictions within this test case."""


@dataclass(frozen=True)
class ClassMetricsPerTestCase(TestCaseMetricsSingleClass):
    """Class-level aggregate metrics for a single class within a multiclass test case."""

    label: str
    """Label associated with this class."""

    nImages: int
    """# of images with this label"""


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    """Test-case-level aggregate metrics for Multiclass Classification workflow."""

    n_labels: int
    """Total number of classes within this test case."""

    n_correct: int
    """Total number of correct predictions within this test case."""

    n_incorrect: int
    """Total number of incorrect predictions within this test case."""

    Accuracy: float
    """Overall accuracy for predictions within this test case. Same as micro-averaged precision, recall and F1-score."""

    macro_Accuracy: float
    """Macro-averaged accuracy for predictions within this test case."""

    macro_Precision: float
    """Macro-averaged precision score for all classes within this test case."""

    macro_Recall: float
    """Macro-averaged recall score for all classes within this test case."""

    macro_F1: float
    """Macro-averaged F1-score for all classes within this test case."""

    macro_FPR: float
    """Macro-averaged false positive rate (FPR) score for all classes within this test case."""

    PerClass: List[ClassMetricsPerTestCase]
    """Class-level metrics for each class within this test case."""


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    """Test-suite-level metrics for Classification workflow."""

    n_images: int
    """Number of images tested within this test suite."""

    n_invalid: int
    """Number of invalid images, i.e. images with no predicted class, within this test suite."""

    n_correct: int
    """Number of correct image classifications within this test suite."""

    overall_accuracy: float
    """Overall accuracy of all the images within this test suite."""


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    """
    Confidence threshold configuration for Classification workflow.
    Specify a minimum confidence `threshold` for predictions, or leave unspecified to include all inferences.
    """

    threshold: Optional[float] = None
    """
    Confidence score threshold to apply for predictions. The prediction with highest confidence is always used,
    but can be invalid if the confidence score is below the threshold.
    """

    def display_name(self) -> str:
        if self.threshold is not None:
            return f"Confidence Above Threshold (threshold={self.threshold})"
        return "Max Confidence"
