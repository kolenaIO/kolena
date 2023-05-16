import dataclasses
from typing import Optional

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
    avg_cost: float
    avg_inference_time: float
    avg_wc_input: int
    avg_wc_gt: int
    avg_wc_inf: int
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
    variance_BERT_f1: float
    variance_BLEU: float
    variance_ROUGE_L: float
