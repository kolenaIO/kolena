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
from enum import Enum
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
    metadata: Metadata = Field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    classification: ClassificationLabel


@dataclass(frozen=True)
class Inference(BaseInference):
    inferences: List[ScoredClassificationLabel]


_workflow, TestCase, TestSuite, Model = define_workflow(
    "Prebuilt Multiclass Classification",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    label: Optional[str]
    score: Optional[float]
    margin: Optional[float]
    is_correct: bool


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
