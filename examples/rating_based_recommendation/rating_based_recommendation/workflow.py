# build out the workflow for the FR problem which is already supported in kolena.fr as a pre-built workflow.
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

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import Composite
from kolena.workflow import Text


@dataclass(frozen=True)
class TestSample(Composite):
    user_id: Text
    title: Text
    movie_id: int
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    rating: float


@dataclass(frozen=True)
class Inference(BaseInference):
    pred_rating: float


workflow, TestCase, TestSuite, Model = define_workflow(
    "Rating-Based Recommendation",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    is_correct: bool
    is_TP: bool
    is_FP: bool
    is_FN: bool
    is_TN: bool
    Δ_rating: float


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    RMSE: float  # Root Mean Squared Error
    MAE: float  # Mean Absolute Error
    TP: int
    FP: int
    FN: int
    TN: int
    Accuracy: float
    Precision: float
    Recall: float
    F1: float
    HighRatingFNR: float
    LowRatingFPR: float


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    rating_threshold: float

    def display_name(self) -> str:
        return f"Rating Threshold: {self.rating_threshold}"
