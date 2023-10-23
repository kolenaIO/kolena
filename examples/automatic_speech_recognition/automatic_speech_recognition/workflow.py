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
    WER: float
    MER: float
    WIL: float
    WIP: float
    CER: float

    AvgGTWordCount: int
    AvgInfWordCount: int


@dataclass(frozen=True)
class TestSuiteMetric(MetricsTestSuite):
    Transcriptions: int
    Failures: int
    FailureRate: float
