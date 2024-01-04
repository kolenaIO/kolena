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

from pydantic.dataclasses import dataclass

from kolena.workflow import Composite
from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import Text


@dataclass(frozen=True)
class SentencePair(Composite):
    sentence1: Text
    sentence2: Text
    sentence1_word_count: int
    sentence2_word_count: int
    sentence1_char_length: int
    sentence2_char_length: int
    word_count_diff: int
    char_length_diff: int
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    similarity: float


@dataclass(frozen=True)
class Inference(BaseInference):
    similarity: float


_, TestCase, TestSuite, Model = define_workflow("Semantic Textual Similarity", SentencePair, GroundTruth, Inference)


@dataclass(frozen=True)
class TestSampleMetric(MetricsTestSample):
    error: float  # GroundTruth.similarity - Inference.similarity
    abs_error: float


@dataclass(frozen=True)
class TestCaseMetric(MetricsTestCase):
    PearsonCorr: float
    SpearmanCorr: float
    MAE: float
    RMSE: float
