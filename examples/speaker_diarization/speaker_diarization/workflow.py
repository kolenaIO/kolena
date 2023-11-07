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
from typing import Optional, List

from pydantic.dataclasses import dataclass

from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Inference as BaseInference
from kolena.workflow import Metadata
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite

from kolena.workflow import Audio
from kolena.workflow.annotation import ClassificationLabel, TimeSegment, LabeledTimeSegment

@dataclass(frozen=True)
class TestSample(Audio):
    metadata: Metadata = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class GroundTruth(BaseGroundTruth):
    transcription: List[LabeledTimeSegment]


@dataclass(frozen=True)
class Inference(BaseInference):
    transcription: List[LabeledTimeSegment]


_workflow, TestCase, TestSuite, Model = define_workflow("Speaker Diarization", TestSample, GroundTruth, Inference)


@dataclass(frozen=True)
class TestSampleMetric(MetricsTestSample):
    DiarizationErrorRate: float
    JaccardErrorRate: float
    DiarizationPurity: float
    DiarizationCoverage: float

    DetectionAccuracy: float
    DetectionPrecision: float
    DetectionRecall: float

    IdentificationErrorRate: float
    IdentificationPrecision: float
    IdentificationRecall: float

    WordErrorRate: float
    CharacterErrorRate: float

    IdentificationError: List[TimeSegment]
    MissedSpeechError: List[TimeSegment]


@dataclass(frozen=True)
class TestCaseMetric(MetricsTestCase):
    DiarizationErrorRate: float
    JaccardErrorRate: float
    DiarizationPurity: float
    DiarizationCoverage: float

    DetectionAccuracy: float
    DetectionPrecision: float
    DetectionRecall: float

    IdentificationErrorRate: float
    IdentificationPrecision: float
    IdentificationRecall: float


@dataclass(frozen=True)
class TestSuiteMetric(MetricsTestSuite):
    Diarizations: int