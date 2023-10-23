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
from typing import Optional
from typing import Tuple

import numpy as np
from recommender_system.workflow import RecommendationConfiguration
from recommender_system.workflow import GroundTruth
from recommender_system.workflow import Inference
from recommender_system.workflow import TestCase
from recommender_system.workflow import TestCaseMetrics
from recommender_system.workflow import TestSample
from recommender_system.workflow import TestSampleMetrics
from recommender_system.workflow import TestSuiteMetrics

from kolena.workflow import AxisConfig
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases


def compute_per_sample(
    ground_truth: GroundTruth,
    inference: Inference,
    configuration: RecommendationConfiguration,
) -> TestSampleMetrics:
    is_correct = False
    if inference.rating is None:
        return TestSampleMetrics(is_correct=is_correct, real_rating=ground_truth.rating)

    if (ground_truth.rating >= configuration.rating_threshold) == (inference.rating >= configuration.rating_threshold):
        is_correct = True

    return TestSampleMetrics(is_correct=is_correct, real_rating=ground_truth.rating, predicted_rating=inference.rating)


def compute_test_case_metrics(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
) -> TestCaseMetrics:
    ratings = np.array([tsm.real_rating for tsm in metrics])
    preds = np.array([tsm.predicted_rating for tsm in metrics])

    rmse = np.sqrt(((preds - ratings) ** 2).mean())
    mae = (np.abs(preds - ratings)).mean()

    return TestCaseMetrics(RMSE=rmse, MAE=mae, Precision=0, Recall=0)


def compute_test_case_plots(ground_truths: List[GroundTruth], inferences: List[Inference]) -> Optional[List[Plot]]:
    return []  # tail plot, coverage plot, confusion matrix, ROC, PR curve, ROC/AUC


def compute_test_suite_metrics(
    test_samples: List[TestSample], configuration: RecommendationConfiguration
) -> TestSuiteMetrics:
    return TestSuiteMetrics(average_RMSE=0, average_MAE=0, average_mAP_k=0, average_mAR_k=0)


def evaluate_recommender_system(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: RecommendationConfiguration,
) -> EvaluationResults:
    # compute per-sample metrics for each test sample
    test_sample_metrics = [compute_per_sample(gt, inf, configuration) for gt, inf in zip(ground_truths, inferences)]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    for test_case, ts, gt, inf, tsm in test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics):
        all_test_case_metrics.append((test_case, compute_test_case_metrics(ts, gt, inf, tsm)))
        all_test_case_plots.append((test_case, compute_test_case_plots(gt, inf)))

    test_suite_metrics = compute_test_suite_metrics(test_samples, configuration)

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
        metrics_test_suite=test_suite_metrics,
    )
