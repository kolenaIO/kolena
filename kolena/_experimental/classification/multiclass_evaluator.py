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
import dataclasses
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np

from kolena._experimental.classification import ClassMetricsPerTestCase
from kolena._experimental.classification import GroundTruth
from kolena._experimental.classification import Inference
from kolena._experimental.classification import TestCase
from kolena._experimental.classification import TestCaseMetrics
from kolena._experimental.classification import TestSample
from kolena._experimental.classification import TestSampleMetrics
from kolena._experimental.classification import TestSuiteMetrics
from kolena._experimental.classification import ThresholdConfiguration
from kolena._experimental.classification.utils import compute_test_case_confidence_histograms
from kolena._experimental.classification.utils import compute_test_case_confusion_matrix
from kolena._experimental.classification.utils import compute_test_case_roc_curves
from kolena._experimental.classification.utils import get_histogram_range
from kolena._experimental.classification.utils import metric_bar_plot_by_class
from kolena._utils import log
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.metrics import f1_score as compute_f1_score
from kolena.workflow.metrics import precision as compute_precision
from kolena.workflow.metrics import recall as compute_recall


def compute_test_sample_metric(
    ground_truth: GroundTruth,
    inference: Inference,
    threshold_configuration: ThresholdConfiguration,
) -> TestSampleMetrics:
    empty_metrics = TestSampleMetrics(
        label=None,
        score=None,
        margin=None,
        is_correct=False,
    )

    if threshold_configuration is None or len(inference.inferences) == 0:
        return empty_metrics

    sorted_infs = sorted(inference.inferences, key=lambda x: x.score, reverse=True)
    predicted_match = sorted_infs[0]
    predicted_label, predicted_score = predicted_match.label, predicted_match.score

    if predicted_score < threshold_configuration.threshold:
        return empty_metrics

    return TestSampleMetrics(
        label=predicted_label,
        score=predicted_score,
        margin=predicted_score - sorted_infs[1].score if len(sorted_infs) >= 2 else None,
        is_correct=predicted_label == ground_truth.classification.label,
    )


def compute_test_case_plots(
    per_class_metrics: List[ClassMetricsPerTestCase],
    gt_labels: List[str],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
    confidence_range: Optional[Tuple[float, float, int]],
) -> List[Plot]:
    plots: List[Plot] = []

    if len(gt_labels) > 1:
        plots = [
            metric_bar_plot_by_class(custom_metric, per_class_metrics)
            for custom_metric in ["Precision", "Recall", "F1", "accuracy"]
        ]

    if confidence_range:
        plots.extend(compute_test_case_confidence_histograms(metrics, confidence_range))
    else:
        log.warn("skipping test case confidence histograms: unsupported confidence range")

    plots.append(compute_test_case_roc_curves(gt_labels, ground_truths, inferences))
    plots.append(compute_test_case_confusion_matrix(ground_truths, metrics))
    plots = list(filter(lambda plot: plot is not None, plots))

    return plots


def compute_test_case_metrics(
    ground_truths: List[GroundTruth],
    metrics_test_samples: List[TestSampleMetrics],
    labels: List[str],
) -> TestCaseMetrics:
    classification_pairs = [
        (gt.classification.label, tsm.classification.label) for gt, tsm in zip(ground_truths, metrics_test_samples)
    ]
    n_images = len(classification_pairs)
    class_level_metrics: List[ClassMetricsPerTestCase] = []
    for label in sorted(labels):
        n_tp = len([True for gt, inf in classification_pairs if gt == label and inf == label])
        n_fn = len([True for gt, inf in classification_pairs if gt == label and inf != label])
        n_fp = len([True for gt, inf in classification_pairs if gt != label and inf == label])
        n_tn = len([True for gt, inf in classification_pairs if gt != label and inf != label])
        n_correct = n_tp + n_tn
        n_incorrect = n_fn + n_fp

        class_level_metrics.append(
            ClassMetricsPerTestCase(
                label=label,
                n_correct=n_correct,
                n_incorrect=n_incorrect,
                accuracy=n_correct / n_images if n_images > 0 else 0,
                Precision=compute_precision(n_tp, n_fp),
                Recall=compute_recall(n_tp, n_fn),
                F1=compute_f1_score(n_tp, n_fp, n_fn),
                FPR=n_fp / (n_fp + n_tn) if n_fp + n_tn > 0 else 0,
            ),
        )

    return TestCaseMetrics(
        PerClass=class_level_metrics,
        n_labels=len(labels),
        n_correct=sum([class_metric.n_correct for class_metric in class_level_metrics]),
        n_incorrect=sum([class_metric.n_incorrect for class_metric in class_level_metrics]),
        macro_accuracy=np.mean([class_metric.Accuracy for class_metric in class_level_metrics]),
        macro_Precision=np.mean([class_metric.Precision for class_metric in class_level_metrics]),
        macro_Recall=np.mean([class_metric.Recall for class_metric in class_level_metrics]),
        macro_F1=np.mean([class_metric.F1 for class_metric in class_level_metrics]),
        macro_FPR=np.mean([class_metric.FPR for class_metric in class_level_metrics]),
    )


def evaluate_classification(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration = dataclasses.field(default_factory=ThresholdConfiguration),
) -> EvaluationResults:
    metrics_test_sample: List[Tuple[TestSample, TestSampleMetrics]] = [
        (ts, compute_test_sample_metric(gt, inf, configuration))
        for ts, gt, inf in zip(test_samples, ground_truths, inferences)
    ]
    test_sample_metrics: List[TestSampleMetrics] = [mts for _, mts in metrics_test_sample]
    confidence_scores: List[float] = [mts.classification.score for mts in test_sample_metrics if mts.classification]
    confidence_range = get_histogram_range(confidence_scores)

    metrics_test_case: List[Tuple[TestCase, TestCaseMetrics]] = []
    plots_test_case: List[Tuple[TestCase, List[Plot]]] = []
    for tc, tc_samples, tc_gts, tc_infs, tc_metrics in test_cases.iter(
        test_samples,
        ground_truths,
        inferences,
        test_sample_metrics,
    ):
        gt_labels: Set[str] = set()
        for tc_gt in tc_gts:
            gt_labels.add(tc_gt.classification.label)
        test_case_metrics: TestCaseMetrics = compute_test_case_metrics(tc_samples, tc_gts, tc_metrics, gt_labels)
        test_case_plots = compute_test_case_plots(
            test_case_metrics.PerClass,
            sorted(gt_labels),
            tc_gts,
            tc_infs,
            tc_metrics,
            confidence_range,
        )
        metrics_test_case.append((tc, test_case_metrics))
        plots_test_case.append((tc, test_case_plots))

    n_images = len(test_sample_metrics)
    n_correct = len([True for tsm in test_sample_metrics if tsm.is_correct])
    metrics_test_suite = TestSuiteMetrics(
        n_images=n_images,
        n_invalid=len([True for tsm in test_sample_metrics if tsm.classification]),
        n_correct=n_correct,
        overall_accuracy=n_correct / n_images if n_images > 0 else 0,
    )

    return EvaluationResults(
        metrics_test_sample=metrics_test_sample,
        metrics_test_case=metrics_test_case,
        plots_test_case=plots_test_case,
        metrics_test_suite=metrics_test_suite,
    )
