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
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from face_recognition_11.workflow import GroundTruth
from face_recognition_11.workflow import Inference
from face_recognition_11.workflow import ThresholdConfiguration
from face_recognition_11.workflow import TestCase
from face_recognition_11.workflow import TestCaseMetrics
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSampleMetrics
from face_recognition_11.workflow import TestSuite

from kolena.workflow import AxisConfig
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import Evaluator
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow import EvaluationResults


def compute_test_sample_metrics(
    ground_truth: GroundTruth,
    inference: Inference,
    configuration: ThresholdConfiguration,
) -> List[Tuple[TestSample, TestSampleMetrics]]:
    is_match, is_false_match, is_false_non_match = False, False, False

    if inference.similarity is None:
        return TestSampleMetrics(is_match, is_false_match, is_false_non_match)

    if inference.similarity > configuration.threshold:  # match
        is_match = True
    else:  # no match
        is_false_match = not ground_truth.is_same
        is_false_non_match = ground_truth.is_same

    return TestSampleMetrics(is_match, is_false_match, is_false_non_match)


def compute_test_case_metrics(
    ground_truths: List[GroundTruth],
    metrics: List[TestSampleMetrics],
) -> TestCaseMetrics:
    n_genuine_pairs = np.sum([gt.is_same for gt in ground_truths])
    n_imposter_pairs = np.sum([not gt.is_same for gt in ground_truths])
    n_fm = np.sum([metric.is_false_match for metric in metrics])
    n_fnm = np.sum([metric.is_false_non_match for metric in metrics])

    return TestCaseMetrics(
        n_images=len(metrics),
        n_genuine_pairs=n_genuine_pairs,
        n_imposter_pairs=n_imposter_pairs,
        n_fm=n_fm,
        fmr=n_fm / n_imposter_pairs,
        n_fnm=n_fnm,
        fnmr=n_fnm / n_genuine_pairs,
    )


def compute_test_case_plots(
    self,
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
    gt_labels: List[str],
    confidence_range: Optional[Tuple[float, float, int]],
) -> List[Plot]:
    plots = []
    # TODO: existing plots depend on baseline - add in baseline fmr

    # set Test Sample as the first baseline to compute remaining metrics

    # AUC and ROC plots

    return plots


# TODO: change to using evaluate function()
def evaluate_face_recognition_11(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration,
) -> EvaluationResults:
    # compute sample-level metrics for each sample
    test_sample_metrics = [
        compute_test_sample_metrics(gt, inf, configuration) for gt, inf in zip(ground_truths, inferences)
    ]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]] = []
    test_case_plots: List[Tuple[TestCase, List[Plot]]] = []

    for test_case, ts, gt, inf, tsm in test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics):
        test_case_metrics.append((test_case, compute_test_case_metrics(gt, tsm)))
        test_case_plots.append((test_case, compute_test_case_plots(test_case, inferences, gt, inf)))

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=test_case_metrics,
        plots_test_case=test_case_plots,
    )
