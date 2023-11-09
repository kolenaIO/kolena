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
from typing import List
from typing import Tuple

from question_answering.truthful_qa.workflow import GroundTruth
from question_answering.truthful_qa.workflow import Inference
from question_answering.truthful_qa.workflow import TestCase
from question_answering.truthful_qa.workflow import TestCaseMetrics
from question_answering.truthful_qa.workflow import TestSample
from question_answering.truthful_qa.workflow import TestSampleMetrics
from tqdm import tqdm

from kolena.workflow import EvaluationResults
from kolena.workflow import TestCases


def compute_test_sample_metrics(
    gt: GroundTruth,
    inf: Inference,
) -> TestSampleMetrics:
    return TestSampleMetrics()


def compute_test_case_metrics(
    metrics: List[TestSampleMetrics],
) -> TestCaseMetrics:
    return TestCaseMetrics()


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
