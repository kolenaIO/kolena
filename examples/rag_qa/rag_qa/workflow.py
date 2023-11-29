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
from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow import Text

workflow, TestCase, TestSuite, Model = define_workflow("Retrieval Augmented Generation", Text, GroundTruth, Inference)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    """Sample-level metrics for the Question Answering workflow."""

    is_correct: bool
    """An indication of the answer being correct or wrong based on the score of MEAN_METRIC."""

    BERT_prec: float
    """The BERT precision score for this test sample."""

    BERT_rec: float
    """The BERT recall score for this test sample."""

    BERT_f1: float
    """The BERT F1 score for this test sample."""

    MEAN_METRIC: float
    """The average value of BERT F1 and ROUGE1."""

    ROUGE_1: float
    """The ROUGE_1 score for this test sample."""

    ROUGE_2: float
    """The ROUGE_2 score for this test sample."""

    ROUGE_L: float
    """The ROUGE_L score for this test sample."""


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    """Test-case-level aggregate metrics for the Question Answering workflow."""

    n_correct: int
    """Total number of correct predictions within this test case."""

    n_incorrect: int
    """Total number of incorrect predictions within this test case."""

    BERT_prec: float
    """Overall BERT precision score for predictions within this test case."""

    BERT_rec: float
    """Overall BERT recall score for predictions within this test case."""

    BERT_f1: float
    """Overall BERT F1 score for predictions within this test case."""

    MEAN_METRIC: float
    """Overall average for BERT F1 and ROUGE_1 scores within this test case."""

    ROUGE_1: float
    """Overall ROUGE_1 score for predictions within this test case."""

    ROUGE_2: float
    """Overall ROUGE_2 score for predictions within this test case."""

    ROUGE_L: float
    """Overall ROUGE_L score for predictions within this test case."""


@dataclass(frozen=True)
class TestSuiteMetrics(MetricsTestSuite):
    """Test-suite-level metrics for the Question Answering workflow."""

    n_stories: int
    """Number of stories tested within this test suite."""

    n_questions: int
    """Number of question and answer pairs tested within this test suite."""

    n_correct: int
    """Number of questions answered correctly within this test suite."""

    overall_accuracy: float
    """The percentage of questions answered correctly within this test suite."""


@dataclass(frozen=True)
class ThresholdConfiguration(EvaluatorConfiguration):
    """
    Similarity score threshold configuration for the Question Answering workflow.
    Specify a minimum similarity `threshold` for answers to be considered valid, defaulting to 0.5.
    """

    threshold: float = 0.5
    """
    Answers are considered correct when a metric similarity score is above this threshold.
    """

    def display_name(self) -> str:
        return f"Similarity Above Threshold (threshold={self.threshold})"
