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
from typing import Optional

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import TestSample as BaseTestSample
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite


@dataclass(frozen=True)
class TestSample(BaseTestSample):
    """Test sample type for Recommender System Workflow"""

    user_id: int
    movie_id: int
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    rating: float


@dataclass(frozen=True)
class Inference(BaseInference):
    rating: float


workflow, TestCase, TestSuite, Model = define_workflow(
    "Recommender System",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    is_correct: bool
    real_rating: float
    predicted_rating: float


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    RMSE: float  # Root Mean Squared Error
    MAE: float  # Mean Absolute Error
    Precision: float
    Recall: float
    # mAP_k: float  # Mean Average Precision
    # mAR_k: float  # Mean Average Recall


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    average_RMSE: float
    average_MAE: float
    # average_mAP_k: float
    # average_mAR_k: float


@dataclass(frozen=True)
class RecommendationConfiguration(EvaluatorConfiguration):
    rating_threshold: float
    k: int

    def display_name(self) -> str:
        return f"Rating Threshold: {self.rating_threshold} (k={self.k})"
