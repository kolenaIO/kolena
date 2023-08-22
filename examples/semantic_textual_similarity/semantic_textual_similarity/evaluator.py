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
from typing import Union

import numpy as np
import pandas as pd
from semantic_textual_similarity.workflow import GroundTruth
from semantic_textual_similarity.workflow import Inference
from semantic_textual_similarity.workflow import SentencePair
from semantic_textual_similarity.workflow import TestCase
from semantic_textual_similarity.workflow import TestCaseMetric
from semantic_textual_similarity.workflow import TestSampleMetric

from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import Histogram
from kolena.workflow import Plot
from kolena.workflow.evaluator_function import EvaluationResults
from kolena.workflow.evaluator_function import TestCases


def compute_test_sample_metrics(gt: GroundTruth, inf: Inference) -> TestSampleMetric:
    return TestSampleMetric(
        error=gt.similarity - inf.similarity,
        abs_error=abs(gt.similarity - inf.similarity),
    )


def compute_aggregate_metrics(
    test_samples_metrics: List[TestSampleMetric],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> TestCaseMetric:
    gts = pd.Series([gt.similarity for gt in ground_truths])
    infs = pd.Series([inf.similarity for inf in inferences])

    abs_errors = np.array([metric.abs_error for metric in test_samples_metrics])
    abs_errors_squared = np.array([metric.abs_error**2 for metric in test_samples_metrics])

    return TestCaseMetric(
        PearsonCorr=gts.corr(infs, method="pearson"),
        SpearmanCorr=gts.corr(infs, method="spearman"),
        MAE=abs_errors.mean(),
        RMSE=np.sqrt(abs_errors_squared.mean()),
    )


def compute_score_distribution_plot(
    score: str,
    metrics: List[Union[SentencePair, TestSampleMetric]],
    binning_info: Optional[Tuple[float, float, float]] = None,  # start, end, num
) -> Histogram:
    scores = [getattr(m, score) for m in metrics]
    bins = np.linspace(*binning_info)

    hist, _ = np.histogram(scores, bins=bins)
    return Histogram(
        title=f"Distribution of {score}",
        x_label=f"{score}",
        y_label="Count",
        buckets=list(bins),
        frequency=list(hist),
    )


def compute_metric_vs_metric_plot(
    x_metric: str,
    y_metric: str,
    x_values: List[Union[int, float]],
    y_values: List[Union[int, float]],
    binning_info: Optional[Tuple[float, float, float]] = None,  # start, end, num
) -> CurvePlot:
    bins = list(np.linspace(*binning_info))

    bins_centers: List[float] = []
    bins_values: List[float] = []
    for i in range(len(bins) - 1):
        lo, hi = bins[i : i + 2]
        bin_values = [y for y, x in zip(y_values, x_values) if lo <= x < hi]
        if len(bin_values) > 0:
            bins_centers.append(lo + ((hi - lo) / 2))
            bins_values.append(np.mean(bin_values))

    return CurvePlot(
        title=f"{y_metric} vs. {x_metric}",
        x_label=f"{x_metric}",
        y_label=f"{y_metric}",
        curves=[Curve(x=bins_centers, y=bins_values)],
    )


def compute_plots(
    metrics: List[TestSampleMetric],
    test_samples: List[SentencePair],
) -> List[Plot]:
    return [
        compute_score_distribution_plot(
            "error",
            metrics,
            (-1.0, 1.0, 21),
        ),
        compute_score_distribution_plot(
            "word_count_diff",
            test_samples,
            (0, 20, 21),
        ),
        compute_score_distribution_plot(
            "char_length_diff",
            test_samples,
            (0, 140, 71),
        ),
        compute_metric_vs_metric_plot(
            "Difference in Word Counts",
            "Similarity Absolute Error",
            [getattr(ts, "word_count_diff") for ts in test_samples],
            [getattr(m, "abs_error") for m in metrics],
            (0, 20, 11),
        ),
        compute_metric_vs_metric_plot(
            "Difference in Charector Lengths",
            "Similarity Absolute Error",
            [getattr(ts, "char_length_diff") for ts in test_samples],
            [getattr(m, "abs_error") for m in metrics],
            (0, 140, 36),
        ),
    ]


def evaluate_semantic_similarity(
    test_samples: List[SentencePair],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
) -> EvaluationResults:
    print("computing test sample metrics...")
    test_sample_metrics = [compute_test_sample_metrics(gt, inf) for gt, inf in zip(ground_truths, inferences)]

    all_test_case_metrics: List[Tuple[TestCase, TestCaseMetric]] = []
    all_test_case_plots: List[Tuple[TestCase, List[Plot]]] = []
    for test_case, tc_test_samples, tc_gts, tc_infs, tc_ts_metrics in test_cases.iter(
        test_samples,
        ground_truths,
        inferences,
        test_sample_metrics,
    ):
        print(f"computing aggregate metrics for test case '{test_case.name}'...")
        test_case_metrics = compute_aggregate_metrics(tc_ts_metrics, tc_gts, tc_infs)
        all_test_case_metrics.append((test_case, test_case_metrics))

        print(f"computing plots for test case '{test_case.name}'...")
        test_case_plots = compute_plots(tc_ts_metrics, tc_test_samples)
        all_test_case_plots.append((test_case, test_case_plots))

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=all_test_case_metrics,
        plots_test_case=all_test_case_plots,
    )
