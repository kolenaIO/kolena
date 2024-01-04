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
from collections import defaultdict
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from age_estimation.workflow import GroundTruth
from age_estimation.workflow import Inference
from age_estimation.workflow import TestCase
from age_estimation.workflow import TestSample

from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import EvaluationResults
from kolena.workflow import Histogram
from kolena.workflow import MetricsTestCase as BaseMetricsTestCase
from kolena.workflow import MetricsTestSample as BaseMetricsTestSample
from kolena.workflow import Plot
from kolena.workflow import TestCases


@dataclass(frozen=True)
class MetricsTestSample(BaseMetricsTestSample):
    error: Optional[float]  # absolute error
    fail_to_detect: bool = False


@dataclass(frozen=True)
class MetricsTestCase(BaseMetricsTestCase):
    n_infer_fail: int
    mae: Optional[float] = None  # mean absolute error
    rmse: Optional[float] = None  # root mean squared error
    failure_rate_err_gt_5: Optional[float] = None


def compute_test_sample_metrics(ground_truth: GroundTruth, inference: Inference) -> MetricsTestSample:
    return MetricsTestSample(
        error=abs(inference.age - ground_truth.age) if inference.age is not None else None,
        fail_to_detect=inference.age is None,
    )


def compute_test_case_metrics(test_sample_metrics: List[MetricsTestSample]) -> MetricsTestCase:
    num_valid_predictions = sum(not metric.fail_to_detect for metric in test_sample_metrics)
    if num_valid_predictions == 0:
        return MetricsTestCase(n_infer_fail=len(test_sample_metrics))

    abs_errors = np.array([metric.error for metric in test_sample_metrics if metric.error is not None])
    abs_errors_squared = np.array([ae**2 for ae in abs_errors])

    return MetricsTestCase(
        mae=abs_errors.mean(),
        rmse=np.sqrt(abs_errors_squared.mean()),
        n_infer_fail=len(test_sample_metrics) - num_valid_predictions,
        failure_rate_err_gt_5=np.sum(abs_errors > 5) / float(num_valid_predictions),
    )


def compute_test_case_plots(
    ground_truths: List[GroundTruth],
    test_sample_metrics: List[MetricsTestSample],
) -> List[Plot]:
    data = [mts.error for mts in test_sample_metrics if mts.error is not None]
    hist, bins = np.histogram(data, bins=100, range=(0, 10))
    histogram_absolute_error = Histogram(
        title="Distribution of Absolute Error",
        x_label="Absolute Error",
        y_label="Count",
        buckets=list(bins),
        frequency=list(hist),
    )

    mae_data = defaultdict(list)
    for gt, mts in zip(ground_truths, test_sample_metrics):
        if mts.error is not None:
            mae_data[gt.age].append(mts.error)

    sorted_data = dict(sorted(mae_data.items()))
    x = list(sorted_data.keys())
    y = [sum(sorted_data[age]) / float(len(sorted_data[age])) for age in x]
    curve_target_age = CurvePlot(
        title="Mean Absolute Error vs. Target Age",
        x_label="Target Age",
        y_label="Mean Absolute Error",
        curves=[Curve(x=x, y=y)],
    )

    return [histogram_absolute_error, curve_target_age]


def evaluate_age_estimation(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
) -> EvaluationResults:
    # compute sample-level metrics for each sample
    test_sample_metrics = [compute_test_sample_metrics(gt, inf) for gt, inf in zip(ground_truths, inferences)]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, MetricsTestCase]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    for test_case, _, gt, _, tsm in test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics):
        all_test_case_metrics.append((test_case, compute_test_case_metrics(tsm)))
        all_test_case_plots.append((test_case, compute_test_case_plots(gt, tsm)))

    # if desired, compute and add `plots_test_case` and `metrics_test_suite` to this `EvaluationResults`
    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
    )
