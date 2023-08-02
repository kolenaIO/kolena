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
from dataclasses import make_dataclass
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
from classification.workflow import ClassMetricsPerTestCase
from classification.workflow import GroundTruth
from classification.workflow import Inference
from classification.workflow import TestCase
from classification.workflow import TestCaseMetrics
from classification.workflow import TestSample
from classification.workflow import TestSampleMetrics
from classification.workflow import TestSuiteMetrics
from classification.workflow import ThresholdConfiguration

from kolena._experimental.classification.utils import compute_confusion_matrix
from kolena._experimental.classification.utils import compute_roc_curves
from kolena._experimental.classification.utils import create_histogram
from kolena._experimental.classification.utils import get_histogram_range
from kolena._utils import log
from kolena.workflow import EvaluationResults
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.metrics import accuracy as compute_accuracy
from kolena.workflow.metrics import f1_score as compute_f1_score
from kolena.workflow.metrics import precision as compute_precision
from kolena.workflow.metrics import recall as compute_recall
from kolena.workflow.plot import Histogram


def compute_test_sample_metric(
    ground_truth: GroundTruth,
    inference: Inference,
    threshold_configuration: ThresholdConfiguration,
) -> TestSampleMetrics:
    empty_metrics = TestSampleMetrics(
        classification=None,
        margin=None,
        is_correct=False,
    )

    if len(inference.inferences) == 0:
        return empty_metrics

    sorted_infs = sorted(inference.inferences, key=lambda x: x.score, reverse=True)
    predicted_match = sorted_infs[0]
    predicted_label, predicted_score = predicted_match.label, predicted_match.score

    if threshold_configuration.threshold is not None and predicted_score < threshold_configuration.threshold:
        return empty_metrics

    return TestSampleMetrics(
        classification=predicted_match,
        margin=predicted_score - sorted_infs[1].score if len(sorted_infs) >= 2 else None,
        is_correct=predicted_label == ground_truth.classification.label,
    )


def compute_test_case_metrics(
    ground_truths: List[GroundTruth],
    metrics_test_samples: List[TestSampleMetrics],
    labels: List[str],
) -> TestCaseMetrics:
    classification_pairs = [
        (gt.classification.label, tsm.classification.label if tsm.classification else None)
        for gt, tsm in zip(ground_truths, metrics_test_samples)
    ]
    n_images = len(classification_pairs)
    class_level_metrics: List[ClassMetricsPerTestCase] = []
    for label in sorted(labels):
        n_tp = len([True for gt, inf in classification_pairs if gt == label and inf == label])
        n_fn = len([True for gt, inf in classification_pairs if gt == label and inf != label])
        n_fp = len([True for gt, inf in classification_pairs if gt != label and inf == label])
        n_tn = len([True for gt, inf in classification_pairs if gt != label and inf != label])

        class_level_metrics.append(
            ClassMetricsPerTestCase(
                label=label,
                TP=n_tp,
                FP=n_fp,
                FN=n_fn,
                TN=n_tn,
                Accuracy=compute_accuracy(n_tp, n_fp, n_fn, n_tn),
                Precision=compute_precision(n_tp, n_fp),
                Recall=compute_recall(n_tp, n_fn),
                F1=compute_f1_score(n_tp, n_fp, n_fn),
                FPR=n_fp / (n_fp + n_tn) if n_fp + n_tn > 0 else 0,
            ),
        )

    n_correct = sum([mts.is_correct for mts in metrics_test_samples])
    return TestCaseMetrics(
        PerClass=class_level_metrics,
        n_labels=len(labels),
        n_correct=n_correct,
        n_incorrect=n_images - n_correct,
        Accuracy=n_correct / n_images,
        macro_Accuracy=np.mean([class_metric.Accuracy for class_metric in class_level_metrics]),
        macro_Precision=np.mean([class_metric.Precision for class_metric in class_level_metrics]),
        macro_Recall=np.mean([class_metric.Recall for class_metric in class_level_metrics]),
        macro_F1=np.mean([class_metric.F1 for class_metric in class_level_metrics]),
        macro_FPR=np.mean([class_metric.FPR for class_metric in class_level_metrics]),
    )


def compute_test_case_confidence_histograms(
    metrics: List[TestSampleMetrics],
    range: Tuple[float, float, int],
) -> List[Histogram]:
    all = [mts.classification.score for mts in metrics if mts.classification]
    correct = [mts.classification.score for mts in metrics if mts.classification and mts.is_correct]
    incorrect = [mts.classification.score for mts in metrics if mts.classification and not mts.is_correct]

    plots = [
        create_histogram(
            values=all,
            range=range,
            title="Score Distribution (All)",
            x_label="Confidence",
            y_label="Count",
        ),
        create_histogram(
            values=correct,
            range=range,
            title="Score Distribution (Correct)",
            x_label="Confidence",
            y_label="Count",
        ),
        create_histogram(
            values=incorrect,
            range=range,
            title="Score Distribution (Incorrect)",
            x_label="Confidence",
            y_label="Count",
        ),
    ]
    return plots


def compute_test_case_plots(
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
    gt_labels: List[str],
    confidence_range: Optional[Tuple[float, float, int]],
) -> List[Plot]:
    plots: List[Plot] = []

    if confidence_range:
        plots.extend(compute_test_case_confidence_histograms(metrics, confidence_range))
    else:
        log.warn("skipping test case confidence histograms: unsupported confidence range")

    plots.append(
        compute_roc_curves(
            [gt.classification for gt in ground_truths],
            [inf.inferences for inf in inferences],
            gt_labels,
        ),
    )
    plots.append(
        compute_confusion_matrix(
            [gt.classification.label for gt in ground_truths],
            [metric.classification.label if metric.classification is not None else "None" for metric in metrics],
        ),
    )
    plots = list(filter(lambda plot: plot is not None, plots))

    return plots


def compute_test_suite_metrics(
    test_sample_metrics: List[TestSampleMetrics],
    configuration: ThresholdConfiguration,
) -> TestSuiteMetrics:
    n_images = len(test_sample_metrics)
    n_correct = sum([tsm.is_correct for tsm in test_sample_metrics])
    metrics = dict(
        n_images=n_images,
        n_invalid=len([mts for mts in test_sample_metrics if mts.classification is None]),
        n_correct=n_correct,
        overall_accuracy=n_correct / n_images if n_images > 0 else 0,
    )

    if configuration.threshold is not None:
        dc = make_dataclass(
            "ExtendedTestSuiteMetrics",
            bases=(TestSuiteMetrics,),
            fields=[("threshold", float)],
            frozen=True,
        )
        metrics_test_suite = dc(
            **metrics,
            threshold=configuration.threshold,
        )
    else:
        metrics_test_suite = TestSuiteMetrics(
            **metrics,
        )

    return metrics_test_suite


def evaluate_classification(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration,
) -> EvaluationResults:
    test_sample_metrics = [
        compute_test_sample_metric(gt, inf, configuration) for gt, inf in zip(ground_truths, inferences)
    ]

    confidence_scores: List[float] = [
        metric.classification.score for metric in test_sample_metrics if metric.classification
    ]
    confidence_range = get_histogram_range(confidence_scores)

    metrics_test_case: List[Tuple[TestCase, TestCaseMetrics]] = []
    plots_test_case: List[Tuple[TestCase, List[Plot]]] = []
    for tc, _, tc_gts, tc_infs, tc_metrics in test_cases.iter(
        test_samples,
        ground_truths,
        inferences,
        test_sample_metrics,
    ):
        gt_labels = sorted({tc_gt.classification.label for tc_gt in tc_gts})
        test_case_metrics = compute_test_case_metrics(tc_gts, tc_metrics, gt_labels)
        test_case_plots = compute_test_case_plots(
            tc_gts,
            tc_infs,
            tc_metrics,
            gt_labels,
            confidence_range,
        )
        metrics_test_case.append((tc, test_case_metrics))
        plots_test_case.append((tc, test_case_plots))

    metrics_test_suite = compute_test_suite_metrics(test_sample_metrics, configuration)

    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=metrics_test_case,
        plots_test_case=plots_test_case,
        metrics_test_suite=metrics_test_suite,
    )
