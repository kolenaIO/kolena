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
from typing import List
from typing import Tuple

from classification.evaluator_binary import BinaryClassificationEvaluator
from classification.evaluator_multiclass import MulticlassClassificationEvaluator
from classification.workflow import GroundTruth
from classification.workflow import Inference
from classification.workflow import TestCase
from classification.workflow import TestCaseMetrics
from classification.workflow import TestSample
from classification.workflow import ThresholdConfiguration

from kolena._experimental.classification.utils import get_histogram_range
from kolena._utils import log
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases


def evaluate_classification(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration,
) -> EvaluationResults:
    is_binary = all(len(inf.inferences) == 1 for inf in inferences)

    if is_binary:
        log.info("evaluating binary classification test suite")
        evaluator = BinaryClassificationEvaluator()
    else:
        log.info("evaluating multiclass classification test suite")
        evaluator = MulticlassClassificationEvaluator()

    test_sample_metrics = [
        evaluator.compute_test_sample_metrics(gt, inf, configuration) for gt, inf in zip(ground_truths, inferences)
    ]

    confidence_scores: List[float] = [
        metric.classification.score for metric in test_sample_metrics if metric.classification
    ]
    confidence_range = get_histogram_range(confidence_scores)

    gt_labels = sorted({gt.classification.label for gt in ground_truths})

    metrics_test_case: List[Tuple[TestCase, TestCaseMetrics]] = []
    plots_test_case: List[Tuple[TestCase, List[Plot]]] = []
    for tc, _, tc_gts, tc_infs, tc_metrics in test_cases.iter(
        test_samples,
        ground_truths,
        inferences,
        test_sample_metrics,
    ):
        test_case_metrics = evaluator.compute_test_case_metrics(tc_gts, tc_metrics)
        test_case_plots = evaluator.compute_test_case_plots(
            tc_gts,
            tc_infs,
            tc_metrics,
            gt_labels,
            confidence_range,
        )
        metrics_test_case.append((tc, test_case_metrics))
        plots_test_case.append((tc, test_case_plots))

    metrics_test_suite = evaluator.compute_test_suite_metrics(test_sample_metrics, configuration)

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=metrics_test_case,
        plots_test_case=plots_test_case,
        metrics_test_suite=metrics_test_suite,
    )
