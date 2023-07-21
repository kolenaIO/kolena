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
    """Test sample type for the pre-built Classification workflow."""

    metadata: Metadata = dataclasses.field(default_factory=dict)
    """The metadata associated with the test sample."""


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


_workflow, _TestCase, _TestSuite, _Model = define_workflow(
    "Image Classification",
    TestSample,
    GroundTruth,
    Inference,
)
"""Example"""


TestCase = _TestCase
"""
[`TestCase`][kolena.workflow.TestCase] definition bound to the pre-built Classification workflow.

Uses the [`Image`][kolena.workflow.Image] test sample type and the
[`GroundTruth`][kolena._experimental.classification.workflow.GroundTruth] definition specific to this workflow.

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
"""[`TestSuite`][kolena.workflow.TestSuite] definition bound to the pre-built Classification workflow."""


Model = _Model
"""[`Model`][kolena.workflow.Model] definition bound to the pre-built Classification workflow."""


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    """Image-level metrics for the pre-built Classification workflow."""

    classification: Optional[ScoredClassificationLabel]
    """
    The model classification for this image. Empty when no inferences have a sufficient confidence score determined
    by the [`ThresholdConfiguration`][kolena._experimental.classification.workflow.ThresholdConfiguration].
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
    """Class-level aggregate metrics for a single class within a test case."""

    label: str
    """Label associated with this class."""

    n_correct: int
    """Number of correct predictions for this class within the test case."""

    n_incorrect: int
    """Number of incorrect predictions for this class within the test case."""

    Accuracy: float
    """Accuracy for predictions for this class within the test case."""

    Precision: float
    """Precision score for this class within this test case."""

    Recall: float
    """Recall score for this class within this test case. Equivalent to **True Positive Rate** (TPR)."""

    F1: float
    """F1 score for this class within this test case."""

    FPR: float
    """False positive rate (FPR) for this class within this test case."""


@dataclass(frozen=True)
class TestCaseMetricsSingleClass(MetricsTestCase):
    """Test-case-level aggregate metrics for the pre-built (binary) Classification workflow."""

    n_correct: int
    """Total number of correct predictions within this test case."""

    n_incorrect: int
    """Total number of incorrect predictions within this test case."""

    Accuracy: float
    """Accuracy score for predictions within this test case."""

    Precision: float
    """Precision score for predictions within this test case."""

    Recall: float
    """Recall score for predictions within this test case."""
    F1: float
    """F1 score for predictions within this test case."""

    FPR: float
    """FPR score for predictions within this test case."""


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    """Test-case-level aggregate metrics for the pre-built Classification workflow."""

    n_labels: int
    """Total number of classes within this test case."""

    n_correct: int
    """Total number of correct predictions within this test case."""

    n_incorrect: int
    """Total number of incorrect predictions within this test case."""

    macro_Accuracy: float
    """Macro-averaged accuracy for predictions within this test case."""

    macro_Precision: float
    """Macro-averaged precision score for all classes within this test case."""

    macro_Recall: float
    """Macro-averaged recall score for all classes within this test case."""

    macro_F1: float
    """Macro-averaged F1 score for all classes within this test case."""

    macro_FPR: float
    """Macro-averaged false positive rate (FPR) score for all classes within this test case."""

    PerClass: List[ClassMetricsPerTestCase]
    """Class-level metrics for each class within this test case."""


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    """Test-suite-level metrics for the pre-built Classification workflow."""

    n_images: int
    """Number of images tested within this test suite."""

    n_invalid: int
    """Number of invalid images, i.e. images with no predicted class, within this test suite."""

    n_correct: int
    """Number of correct image classifications within this test suite."""

    n_incorrect: int
    """Number of incorrect image classifications within this test suite."""

    overall_accuracy: float
    """Overall accuracy of all the images within this test suite."""


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    """
    Confidence threshold configuration for the pre-built Classification workflow.

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
