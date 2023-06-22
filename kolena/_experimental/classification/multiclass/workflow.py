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

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Image
from kolena.workflow import Inference as BaseInference
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import ScoredClassificationLabel


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for the pre-built Multiclass Classification workflow."""

    classification: ClassificationLabel
    """The classfication label associated with an image."""


@dataclass(frozen=True)
class Inference(BaseInference):
    inferences: List[ScoredClassificationLabel]
    """
    Model predictions for an image. For `N`-class problems, `inferences` is expected to contain `N` entries, one for
    each class and its associated confidence score.
    """


_workflow, _TestCase, _TestSuite, _Model = define_workflow(
    "Multiclass Classification",
    Image,
    GroundTruth,
    Inference,
)
"""Example"""

TestCase = _TestCase
"""
[`TestCase`][kolena.workflow.TestCase] definition bound to the pre-built Multiclass Classification workflow.

Uses the [`Image`][kolena.workflow.Image] test sample type and the
[`GroundTruth`][kolena.classification.multiclass.GroundTruth] definition specific to this workflow.

!!! tip "Tip: Extend `Image` test sample type"

    Any test sample extending [`kolena.workflow.Image`][kolena.workflow.Image] can be used for this workflow. If your
    problem requires additional fields on the test sample, or if you would like to add free-form metadata, simply
    extend this class:

    ```python
    from dataclasses import dataclass, field

    from kolena.workflow import Image, Metadata

    @dataclass(frozen=True)
    class ExampleExtendedImage(Image):
        # locator: str  # inherit from parent Image class
        example_field: str  # add as many fields as desired, including annotations and assets
        example_optional: Optional[int] = None
        metadata: Metadata = field(default_factory=dict)  # optional metadata dict
    ```
"""

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
class PerClassMetrics(MetricsTestCase):
    """Class-level aggregate metrics for a single class within a test case."""

    label: str
    """Label associated with this class."""

    Precision: float
    """Precision score for this class within this test case."""

    Recall: float
    """Recall score for this class within this test case. Equivalent to **True Positive Rate** (TPR)."""

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

    Accuracy: float
    """Accuracy for predictions within this test case."""

    Precision_macro: float
    """Macro-averaged precision score for all classes within this test case."""

    Recall_macro: float
    """Macro-averaged recall score for all classes within this test case."""

    F1_macro: float
    """Macro-averaged F1 score for all classes within this test case."""

    FPR_macro: float
    """Macro-averaged false positive rate (FPR) score for all classes within this test case."""

    PerClass: List[PerClassMetrics]
    """Class-level metrics for all classes within this test case."""


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    """
    Test-suite-level metrics for the pre-built Multiclass Classification workflow.

    In addition to the fixed fields below, these metrics are also populated with class-level `mean_X` and `variance_X`
    for all classes `X`.
    """

    n_images: int
    """Total number of images tested within this test suite."""

    n_images_skipped: int
    """Number of skipped images, i.e. images with no predicted class, within this test suite."""

    variance_Accuracy: float
    """
    Variance of [accuracy][kolena.classification.multiclass.AggregateMetrics.Accuracy] scores across all test cases
    within this test suite.
    """

    variance_Precision_macro: float
    """
    Variance of [macro-averaged precision][kolena.classification.multiclass.AggregateMetrics.Precision_macro] scores
    across all test cases within this test suite.
    """

    variance_Recall_macro: float
    """
    Variance of [macro-averaged recall][kolena.classification.multiclass.AggregateMetrics.Recall_macro] across all test
    cases within this test suite.
    """

    variance_F1_macro: float
    """
    Variance of [macro-averaged F1 scores][kolena.classification.multiclass.AggregateMetrics.F1_macro] across all test
    cases within this test suite.
    """

    variance_FPR_macro: float
    """
    Variance of [macro-averaged false positives rates][kolena.classification.multiclass.AggregateMetrics.FPR_macro]
    across all test cases within this test suite.
    """


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
