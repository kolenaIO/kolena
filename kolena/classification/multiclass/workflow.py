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


@dataclass(frozen=True)
class TestSample(Image):
    metadata: Metadata = Field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    classification: ClassificationLabel


@dataclass(frozen=True)
class InferenceLabel(ClassificationLabel):
    confidence: float


@dataclass(frozen=True)
class Inference(BaseInference):
    inferences: List[InferenceLabel]


_workflow, TestCase, TestSuite, Model = define_workflow("Multiclass Classification", TestSample, GroundTruth, Inference)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    classification: Optional[InferenceLabel]
    margin: Optional[float]
    is_correct: bool


@dataclass(frozen=True)
class AggregatedMetrics:
    Count: int
    F1: float
    Precision: float
    Recall: float
    TPR: float
    FPR: float


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    n_correct: int
    accuracy: float


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    n_images: int
    n_correct: int
    mean_test_case_accuracy: float


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    threshold: Optional[float] = None

    def display_name(self) -> str:
        if self.threshold:
            return f"Confidence Above Threshold (threshold={self.threshold})"
        return "Max Confidence"
