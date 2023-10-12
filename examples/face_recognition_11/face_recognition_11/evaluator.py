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
from face_recognition_11.workflow import FMRConfiguration
from face_recognition_11.workflow import GroundTruth
from face_recognition_11.workflow import Inference
from face_recognition_11.workflow import TestCase
from face_recognition_11.workflow import TestCaseMetrics
from face_recognition_11.workflow import TestSample
from face_recognition_11.workflow import TestSampleMetrics
from face_recognition_11.workflow import TestSuiteMetrics

from kolena.workflow import AxisConfig
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import EvaluationResults
from kolena.workflow import Histogram
from kolena.workflow import Plot
from kolena.workflow import TestCases


def compute_threshold(inferences: List[Tuple[GroundTruth, Inference]], fmr: float) -> float:
    """
    The threhsold is calculated as Threshold = Quantile_v(1 - FMR) where the quantiles are of the observed
    IMPOSTER scores, v.

    For more details on how threshold is used see https://pages.nist.gov/frvt/reports/11/frvt_11_report.pdf
    """
    similarity_scores = np.array(
        [inf.similarity if inf.similarity is not None else 0.0 for gt, inf in inferences if not gt.is_same],
    )
    threshold = np.quantile(similarity_scores, 1.0 - fmr)
    return threshold


def compute_per_sample(
    ground_truth: GroundTruth,
    inference: Inference,
    threshold: float,
) -> TestSampleMetrics:
    is_match, is_false_match, is_false_non_match = False, False, False

    if inference.similarity is None:
        return TestSampleMetrics(
            is_match=is_match,
            is_false_match=is_false_match,
            is_false_non_match=is_false_non_match,
            failure_to_enroll=True,
        )

    if inference.similarity > threshold:  # match
        if ground_truth.is_same:
            is_match = True
        else:
            is_false_match = True
    else:  # no match
        if ground_truth.is_same:
            is_false_non_match = True

    return TestSampleMetrics(
        is_match=is_match,
        is_false_match=is_false_match,
        is_false_non_match=is_false_non_match,
        failure_to_enroll=False,
    )


def compute_test_case_metrics(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
    baseline_fnmr: float,
) -> TestCaseMetrics:
    n_genuine_pairs = np.sum([gt.is_same for gt in ground_truths])
    n_imposter_pairs = np.sum([not gt.is_same for gt in ground_truths])
    N = n_genuine_pairs + n_imposter_pairs

    n_fm = np.sum([metric.is_false_match for metric in metrics])
    n_fnm = np.sum([metric.is_false_non_match for metric in metrics])
    n_pair_failures = np.sum([metric.failure_to_enroll for metric in metrics])
    n_fte = np.sum(
        [int(len(inf.a_keypoints.points) == 0) + int(len(inf.b_keypoints.points) == 0) for inf in inferences],
    )

    unique_images = set()
    for ts in test_samples:
        unique_images.add(ts.a.locator)
        unique_images.add(ts.b.locator)

    Δ_fnmr = 0.0
    if baseline_fnmr != 0.0:
        Δ_fnmr = (n_fnm / n_genuine_pairs) * 100 - baseline_fnmr

    return TestCaseMetrics(
        n_images=len(unique_images),
        n_genuine_pairs=n_genuine_pairs,
        n_imposter_pairs=n_imposter_pairs,
        n_fm=n_fm,
        fmr=(n_fm / n_imposter_pairs) * 100,
        n_fnm=n_fnm,
        fnmr=(n_fnm / n_genuine_pairs) * 100,
        Δ_fnmr=Δ_fnmr,
        n_fte=n_fte,
        fter=(n_fte / N) * 100,
        n_pair_failures=n_pair_failures,
        pair_failure_rate=(n_pair_failures / N) * 100,
    )


def compute_test_case_plots(ground_truths: List[GroundTruth], inferences: List[Inference]) -> Optional[List[Plot]]:
    FMR_lower = -6
    FMR_upper = -1
    baseline_fmr_x = list(np.logspace(FMR_lower, FMR_upper, 50))

    thresholds = [compute_threshold(zip(ground_truths, inferences), fmr) for fmr in baseline_fmr_x]

    n_genuine_pairs = np.sum([gt.is_same for gt in ground_truths])
    n_imposter_pairs = np.sum([not gt.is_same for gt in ground_truths])

    fnmr_y = list()
    fmr_y = list()

    for threshold in thresholds:
        tsm_for_one_threshold = [compute_per_sample(gt, inf, threshold) for gt, inf in zip(ground_truths, inferences)]
        n_fm = np.sum([metric.is_false_match and not metric.failure_to_enroll for metric in tsm_for_one_threshold])
        n_fnm = np.sum(
            [metric.is_false_non_match and not metric.failure_to_enroll for metric in tsm_for_one_threshold],
        )
        fnmr_y.append((n_fnm / n_genuine_pairs) * 100)
        fmr_y.append(n_fm / n_imposter_pairs)

    curve_test_case_fnmr = CurvePlot(
        title="Test Case FNMR vs. Baseline FMR",
        x_label="Baseline False Match Rate",
        y_label="Test Case False Non-Match Rate (%)",
        x_config=AxisConfig(type="log"),
        curves=[Curve(x=baseline_fmr_x, y=fnmr_y, extra=dict(Threshold=thresholds))],
    )

    curve_test_case_fmr = CurvePlot(
        title="Test Case FMR vs. Baseline FMR",
        x_label="Baseline False Match Rate",
        y_label="Test Case False Match Rate",
        x_config=AxisConfig(type="log"),
        y_config=AxisConfig(type="log"),
        curves=[Curve(x=baseline_fmr_x, y=fmr_y, extra=dict(Threshold=thresholds))],
    )

    genuine_values = [
        inf.similarity for gt, inf in zip(ground_truths, inferences) if gt.is_same and inf.similarity is not None
    ]

    imposter_values = [
        inf.similarity for gt, inf in zip(ground_truths, inferences) if not gt.is_same and inf.similarity is not None
    ]

    min_data = min(min(genuine_values), min(imposter_values))
    max_data = max(max(genuine_values), max(imposter_values))

    number_of_bins = 50
    bin_size = (max_data - min_data) / number_of_bins
    bin_edges = [min_data + i * bin_size for i in range(number_of_bins + 1)]

    hist1_adjusted = list(np.histogram(genuine_values, bins=bin_edges, density=True)[0])
    hist2_adjusted = list(np.histogram(imposter_values, bins=bin_edges, density=True)[0])

    # histogram of the relative distribution of genuine and imposter pairs, bucketed by similarity score.
    similarity_dist = Histogram(
        title="Similarity Distribution",
        x_label="Similarity Score",
        y_label="Frequency (%)",
        buckets=list(bin_edges),
        frequency=list([hist1_adjusted, hist2_adjusted]),
        labels=["Genuine Pairs", "Imposter Pairs"],
    )

    return [similarity_dist, curve_test_case_fnmr, curve_test_case_fmr]


def compute_test_suite_metrics(
    test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]],
    threshold: float,
) -> TestSuiteMetrics:
    baseline: TestCaseMetrics = test_case_metrics[0][1]

    return TestSuiteMetrics(
        threshold=threshold,
        n_fm=baseline.n_fm,
        n_fnm=baseline.n_fnm,
        fnmr=baseline.n_fnm / baseline.n_genuine_pairs,
    )


def evaluate_face_recognition_11(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: FMRConfiguration,
) -> EvaluationResults:
    threshold = compute_threshold(zip(ground_truths, inferences), configuration.false_match_rate)

    # compute per-sample metrics for each test sample
    test_sample_metrics = [compute_per_sample(gt, inf, threshold) for gt, inf in zip(ground_truths, inferences)]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetrics]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    baseline_fnmr = 0.0
    for test_case, ts, gt, inf, tsm in test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics):
        all_test_case_metrics.append((test_case, compute_test_case_metrics(ts, gt, inf, tsm, baseline_fnmr)))
        all_test_case_plots.append((test_case, compute_test_case_plots(gt, inf)))

        # first processed test case is baseline
        if len(all_test_case_metrics) == 1:
            baseline_fnmr = all_test_case_metrics[0][1].fnmr

    test_suite_metrics = compute_test_suite_metrics(all_test_case_metrics, threshold)

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
        metrics_test_suite=test_suite_metrics,
    )
