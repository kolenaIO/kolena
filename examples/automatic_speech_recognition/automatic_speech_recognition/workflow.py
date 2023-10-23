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

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow import Audio
from kolena.workflow.annotation import ClassificationLabel


@dataclass(frozen=True)
class TestSample(Audio):
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    transcription: ClassificationLabel


@dataclass(frozen=True)
class Inference(BaseInference):
    transcription: ClassificationLabel # TODO

_workflow, TestCase, TestSuite, Model = define_workflow("Speech Recognition", TestSample, GroundTruth, Inference)


@dataclass(frozen=True)
class TestSampleMetric(MetricsTestSample):
    WordErrors: float
    WordErrorRate: float
    MatchErrorRate: float
    WordInformationLost: float
    WordInformationPreserved: float
    CharacterErrorRate: float

    FNCount: int
    FPCount: int
    InsertionCount: int
    DeletionCount: int
    SubstitutionCount: int

    FalseNegativeText: str
    FalsePositiveText: str

    Substitutions: List[ClassificationLabel]
    Insertions: List[ClassificationLabel]
    Deletions: List[ClassificationLabel]

    Language: str


@dataclass(frozen=True)
class TestCaseMetric(MetricsTestCase):
    FailCount: int
    FailRate: float

    AvgWordErrors: float
    WordErrorRate: float
    MatchErrorRate: float
    WordInfoLost: float
    WordInfoPreserved: float
    CharacterErrorRate: float

    AvgGTWordCount: int
    AvgInfWordCount: int


@dataclass(frozen=True)
class TestSuiteMetric(MetricsTestSuite):
    Transcriptions: int
    Failures: int
    FailureRate: float
