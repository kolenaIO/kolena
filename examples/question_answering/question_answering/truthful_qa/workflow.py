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

    answers: List[Answer]
    num_answers: int
    missing_answer: bool


workflow, TestCase, TestSuite, Model = define_workflow(
    "Question Answering [open-domain]",
    TestSample,
    GroundTruth,
    Inference,
)


@dataclass(frozen=True)
class AnswerResult(Label):
    """Metrics for each answer from a Question Answering model."""

    BART: float
    BERT_prec: float
    BERT_rec: float
    BERT_f1: float
    BLEURT: float
    METEOR: float


@dataclass(frozen=True)
class TestSampleMetrics(MetricsTestSample):
    """Sample-level metrics for the Question Answering workflow."""

    fail_to_answer: bool
    answers: List[AnswerResult]
    best_answer_by_BART: Optional[AnswerResult]
    best_answer_by_BERT_f1: Optional[AnswerResult]
    best_answer_by_BLEURT: Optional[AnswerResult]
    best_answer_by_METEOR: Optional[AnswerResult]
    best_overall: Optional[AnswerResult]


@dataclass(frozen=True)
class TestCaseMetrics(MetricsTestCase):
    """Test-case-level aggregate metrics for the Question Answering workflow."""

    Questions: int
    Failures: int
    BART: float  # from overall_best
    BERT_f1: float  # from overall_best
    BLEURT: float  # from overall_best
    METEOR: float  # from overall_best
