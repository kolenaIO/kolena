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
from typing import Dict
from typing import List
from typing import Optional

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import Text
from kolena.workflow.annotation import Label
from kolena.workflow.asset import PlainTextAsset


@dataclass(frozen=True)
class TestSample(Text):
    """Test sample type for the Question Answering workflow."""

    question: str
    context: Optional[List[str]]
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    """Ground truth type for the Question Answering workflow."""

    best_answer: str
    answers: List[str]
    incorrect_answers: List[str]


@dataclass(frozen=True)
class Answer(Label):
    """Answer prediction from a Question Answering model."""

    raw: str
    text_offset: List[int]
    logprobs: List[float]
    tokens: List[str]
    top_logprobs: List[Dict[str, float]]
    finish_reason: str
    completion_tokens: Optional[int]
    prompt_tokens: Optional[int]
    total_tokens: Optional[int]


@dataclass(frozen=True)
class Inference(BaseInference):
    """Inference type for the Question Answering workflow."""

    missing_answer: bool
    answers: List[Label]
    answer: Optional[Label] = None
    answer_with_top5_logprob: Optional[Label] = None
    selfcheck_metrics: Optional[PlainTextAsset] = None
    probabilities_metrics: Optional[PlainTextAsset] = None
    consistency_metrics: Optional[PlainTextAsset] = None
    is_hallucination: Optional[bool] = None


workflow, TestCase, TestSuite, Model = define_workflow(
    "Question Answering [open-domain]",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    """Sample-level metrics for the Question Answering workflow."""

    fail_to_answer: bool
    is_hallucination_by_logprob: Optional[bool] = None
    is_hallucination_by_entropy: Optional[bool] = None
    average_logprob: Optional[float] = None  # 1 - Avg(logP)
    average_entropy: Optional[float] = None  # Avg(H)
    min_logprob: Optional[float] = None  # 1 - Max(-logP)
    max_entropy: Optional[float] = None  # Max(H)
    is_hallucination_by_selfcheck_bertscore: Optional[bool] = None
    is_hallucination_by_selfcheck_ngram: Optional[bool] = None
    selfcheck_bertscore: Optional[float] = None
    selfcheck_ngram: Optional[float] = None
    is_hallucination_by_selfcheck_prompt: Optional[bool] = None
    selfcheck_prompt: Optional[float] = None
    selfcheck_prompt_reasons: Optional[List[str]] = None


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    """Test-case-level aggregate metrics for the Question Answering workflow."""

    Questions: int
    Failures: int
    FactualityScoreLogProb: float
    FactualityScoreEntropy: float
    FactualityScoreSelfcheckBert: float
    FactualityScoreSelfcheckNGram: float
    MetricsAccuracyLogProb: float
    MetricsAccuracyEntropy: float
    MetricsAccuracySelfcheckBert: float
    MetricsAccuracySelfcheckNGram: float
    MetricsAccuracySelfcheckPrompt: float
