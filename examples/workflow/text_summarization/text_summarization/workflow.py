# Copyright 2021-2024 Kolena Inc.
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
from typing import Union

import numpy as np
from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow import Text


@dataclass(frozen=True)
class TestSample(Text):
    id: str
    word_count: int
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    summary: str
    word_count: int


@dataclass(frozen=True)
class Inference(BaseInference):
    summary: str
    word_count: int
    is_failure: bool = False
    inference_time: Optional[float] = None
    tokens_input_text: Optional[int] = None
    tokens_ref_summary: Optional[int] = None
    tokens_pred_summary: Optional[int] = None
    tokens_prompt: Optional[int] = None
    cost: Optional[float] = None


_workflow, TestCase, TestSuite, Model = define_workflow("Text Summarization", TestSample, GroundTruth, Inference)


@dataclass(frozen=True)
class TestSampleMetric(MetricsTestSample):
    BERT_prec: float
    BERT_rec: float
    BERT_f1: float
    ROUGE_1: float
    ROUGE_2: float
    ROUGE_L: float
    BLEU: float
    inf_to_gt_word_count: float


@dataclass(frozen=True)
class TestCaseMetric(MetricsTestCase):
    n_failures: int
    failure_rate: float
    total_cost: float
    avg_cost: Union[float, np.floating]
    avg_inference_time: Union[float, np.floating]
    avg_wc_input: Union[float, np.floating]
    avg_wc_gt: Union[float, np.floating]
    avg_wc_inf: Union[float, np.floating]
    inf_to_gt_word_count: float
    BERT_prec: float
    BERT_rec: float
    BERT_f1: float
    ROUGE_1: float
    ROUGE_2: float
    ROUGE_L: float
    BLEU: float


@dataclass(frozen=True)
class TestSuiteMetric(MetricsTestSuite):
    num_articles: int
    num_failures: int
    failure_rate: float
    total_cost: float
    variance_BERT_f1: Union[float, np.floating]
    variance_BLEU: Union[float, np.floating]
    variance_ROUGE_L: Union[float, np.floating]
