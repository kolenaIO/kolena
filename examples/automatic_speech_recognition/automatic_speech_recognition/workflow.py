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
    # fp: List[ClassificationLabel]
    # fn: List[ClassificationLabel]
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
    word_errors: float
    word_error_rate: float
    match_error_rate: float
    word_information_lost: float
    word_information_preserved: float
    character_error_rate: float

    word_fn: str
    word_fp: str

    fn_count: int
    fp_count: int

    language: str


@dataclass(frozen=True)
class TestCaseMetric(MetricsTestCase):
    n_failures: int
    failure_rate: float

    avg_word_errors: float
    avg_match_error_rate: float
    avg_word_error_rate: float
    avg_word_information_lost: float
    avg_word_information_preserved: float
    avg_character_error_rate: float

    avg_wc_gt: int
    avg_wc_inf: int


@dataclass(frozen=True)
class TestSuiteMetric(MetricsTestSuite):
    num_transcriptions: int
    num_failures: int
    failure_rate: float
    variance_word_error_rate: float
    variance_match_error_rate: float
    variance_word_information_lost: float
    variance_word_information_preserved: float
    variance_character_error_rate: float