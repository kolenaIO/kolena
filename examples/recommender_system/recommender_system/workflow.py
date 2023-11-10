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
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import TestSample as BaseTestSample
from kolena.workflow.annotation import ScoredClassificationLabel


@dataclass(frozen=True)
class Movie(ScoredClassificationLabel):
    """A User-Movie pair denoting details about the movie and the users associated rating."""

    label: Optional[str]  # movie title
    score: int  # rating
    id: int
    metadata: Optional[Metadata] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class TestSample(BaseTestSample):
    user_id: int
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    rated_movies: List[Movie]


@dataclass(frozen=True)
class Inference(BaseInference):
    recommendations: List[Movie]


workflow, TestCase, TestSuite, Model = define_workflow(
    "Recommender System",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    RMSE: float
    MAE: float
    R2: float
    AP: float
    MAP: float
    MRR: float
    NDCG: float
    F1_k: float
    precision_k: float
    recall_k: float


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    """Test Case level metrics"""

    # Hit Metrics
    AvgRMSE: float  # Root Mean Squared Error
    AvgMAE: float  # Mean Absolute Error
    AvgR2: float  # R Squared

    # Ranking Metrics
    AvgAP: float
    AvgMAP: float  # Mean Average Precision
    AvgMRR: float  # Mean Reciprocal Rank
    AvgNDCG: float  # Normalized Discounted Cumulative Gain
    AvgPrecision_k: float  # Average Precision@k is the mean of each user P@k
    AvgRecall_k: float  # Average Recall@k is the mean of each user R@k


@dataclass(frozen=True)
class RecommenderConfiguration(EvaluatorConfiguration):
    k: int
    """Number of items recommended to the user."""

    relevancy_method: Union[Literal["Top-K"], Literal["Timestep"]] = "Top-K"
    """The method to use to extract relevant items from the recommendation list."""

    def display_name(self) -> str:
        if self.revelancy_method == "Top-K":
            return f"Relevancy Method: Top-K (k={self.k})"

        return f"Relevancy Method: {self.revelancy_method}"
