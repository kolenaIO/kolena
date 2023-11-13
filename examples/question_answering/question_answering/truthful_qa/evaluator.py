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
from operator import attrgetter
from typing import List
from typing import Tuple

import evaluate
import numpy as np
from bart_score import BARTScorer
from question_answering.truthful_qa.workflow import AnswerResult
from question_answering.truthful_qa.workflow import GroundTruth
from question_answering.truthful_qa.workflow import Inference
from question_answering.truthful_qa.workflow import TestCase
from question_answering.truthful_qa.workflow import TestCaseMetrics
from question_answering.truthful_qa.workflow import TestSample
from question_answering.truthful_qa.workflow import TestSampleMetrics
from question_answering.utils import mean_metric
from tqdm import tqdm

from kolena.workflow import EvaluationResults
from kolena.workflow import TestCases


bart_scorer = BARTScorer(device="cuda:0", checkpoint="facebook/bart-large-cnn")
bert_scorer = evaluate.load("bertscore")
bleurt_scorer = evaluate.load("bleurt", module_type="metric")
meteor_scorer = evaluate.load("meteor")


def compute_metrics(gt: str, inf: str) -> AnswerResult:
    bertscore_results = bert_scorer.compute(
        predictions=[inf],
        references=[gt],
        lang="en",
        model_type="distilbert-base-uncased",
    )

    return AnswerResult(
        BART=bart_scorer.score([inf], [gt]),
        BERT_prec=bertscore_results["precision"][0],
        BERT_rec=bertscore_results["recall"][0],
        BERT_f1=bertscore_results["f1"][0],
        BLEURT=bleurt_scorer.compute([inf], [gt])["scores"][0],
        METEOR=meteor_scorer.compute([inf], [gt])["meteor"],
    )


def compute_test_sample_metrics(
    gt: GroundTruth,
    inf: Inference,
) -> TestSampleMetrics:
    answers = [compute_metrics(gt.best_answer, inf_answer) for inf_answer in inf.answers]

    best_answers = dict()
    for metric in ["BART", "BERT_f1", "BLEURT", "METEOR"]:
        best_answers[metric] = max(answers, key=attrgetter(metric))

    best_overall = max(set(best_answers.values()), key=best_answers.count)

    if len(answers) == 0:
        return TestSampleMetrics(
            fail_to_answer=True,
            answers=[],
        )

    return TestSampleMetrics(
        fail_to_answer=False,
        answers=answers,
        best_answer_by_BART=best_answers["BART"],
        best_answer_by_BERT_f1=best_answers["BERT_f1"],
        best_answer_by_BLEURT=best_answers["BLEURT"],
        best_answer_by_METEOR=best_answers["METEOR"],
        best_overall=best_overall,
    )


def compute_test_case_metrics(
    metrics: List[TestSampleMetrics],
) -> TestCaseMetrics:
    return TestCaseMetrics(
        Questions=len(metrics),
        Failures=np.sum([m.fail_to_answer for m in metrics]),
        BART=mean_metric("BART", metrics),
        BERT_f1=mean_metric("BERT_f1", metrics),
        BLEURT=mean_metric("BLEURT", metrics),
        METEOR=mean_metric("METEOR", metrics),
    )


def evaluate_question_answering(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
) -> EvaluationResults:
    # compute sample-level metrics for each sample
    test_sample_metrics = [
        compute_test_sample_metrics(gt, inf)
        for gt, inf in tqdm(zip(ground_truths, inferences), total=len(ground_truths))
    ]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]] = []
    for test_case, ts, gt, inf, tsm in test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics):
        all_test_case_metrics.append((test_case, compute_test_case_metrics(tsm)))

    # if desired, compute and add `plots_test_case` and `metrics_test_suite` to this `EvaluationResults`
    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
    )
