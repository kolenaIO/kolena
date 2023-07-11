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
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple

import numpy as np

from kolena._experimental.classification import GroundTruth
from kolena._experimental.classification import Inference
from kolena._experimental.classification import TestCase
from kolena._experimental.classification import TestSample
from kolena._experimental.classification import ThresholdConfiguration
from kolena._experimental.classification.workflow import ClassMetricsPerTestCase
from kolena._experimental.classification.workflow import TestCaseMetrics
from kolena._experimental.classification.workflow import TestSampleMetrics
from kolena._experimental.classification.workflow import TestSuiteMetrics
from kolena._utils import log
from kolena.classification.multiclass._utils import get_histogram_range
from kolena.classification.multiclass._utils import get_label_confidence
from kolena.classification.multiclass._utils import roc_curve
from kolena.workflow import BarPlot
from kolena.workflow import ConfusionMatrix
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import EvaluationResults
from kolena.workflow import Histogram
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.metrics import f1_score as compute_f1_score
from kolena.workflow.metrics import precision as compute_precision
from kolena.workflow.metrics import recall as compute_recall

Result = Tuple[TestSample, GroundTruth, Inference]


def _compute_test_sample_metric(
    threshold_configuration: ThresholdConfiguration,
    ground_truth: GroundTruth,
    inference: Inference,
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

    if predicted_score < threshold_configuration.min_confidence_score:
        return empty_metrics

    return TestSampleMetrics(
        label=predicted_label,
        score=predicted_score,
        margin=predicted_score - sorted_infs[1].score if len(sorted_infs) >= 2 else None,
        is_correct=predicted_label == ground_truth.classification.label,
    )


def _as_class_metric_plot(
    metric_name: str,
    per_class_metrics: List[ClassMetricsPerTestCase],
    labels: List[str],
) -> Optional[BarPlot]:
    if metric_name == "Recall":
        title = f"{metric_name} (TPR) vs. Class"
    else:
        title = f"{metric_name} vs. Class"

    values = [(pcm.label, getattr(pcm, metric_name)) for pcm in per_class_metrics]
    valid_pairs = [(label, value) for label, value in values if value != 0.0]
    if len(valid_pairs) > 0:
        return BarPlot(
            title=title,
            x_label="Class",
            y_label=metric_name,
            labels=[label for label, _ in valid_pairs],
            values=[value for _, value in valid_pairs],
        )
    return None


def _as_confidence_histogram(
    title: str,
    confidence_scores: List[float],
    confidence_range: Tuple[float, float, int] = (0, 1, 25),
) -> Histogram:
    min_range, max_range, bins = confidence_range
    hist, bins = np.histogram(
        confidence_scores,
        bins=bins,
        range=(min_range, max_range),
    )
    return Histogram(
        title=title,
        x_label="Confidence",
        y_label="Count",
        buckets=list(bins),
        frequency=list(hist),
    )


def _compute_confidence_histograms(
    test_case_name: str,
    metrics: List[TestSampleMetrics],
    confidence_range: Optional[Tuple[float, float, int]],
) -> List[Histogram]:
    if confidence_range is None:
        log.warn(
            f"skipping confidence histograms for {test_case_name}: unsupported confidence range",
        )
        return []

    confidence_all = [mts.score for mts in metrics if mts.label is not None]
    confidence_correct = [mts.score for mts in metrics if mts.label is not None and mts.is_correct]
    confidence_incorrect = [mts.score for mts in metrics if mts.label is not None and not mts.is_correct]

    plots = [
        _as_confidence_histogram("Score Distribution (All)", confidence_all, confidence_range),
        _as_confidence_histogram("Score Distribution (Correct)", confidence_correct, confidence_range),
        _as_confidence_histogram("Score Distribution (Incorrect)", confidence_incorrect, confidence_range),
    ]
    return plots


def _compute_test_case_plots(
    test_case_name: str,
    per_class_metrics: List[ClassMetricsPerTestCase],
    labels: List[str],
    gt_labels: List[str],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[TestSampleMetrics],
    confidence_range: Optional[Tuple[float, float, int]],
) -> List[Plot]:
    plots: List[Plot] = []

    if len(gt_labels) > 1:
        plots = [
            _as_class_metric_plot(custom_metric, per_class_metrics, labels)
            for custom_metric in ["Precision", "Recall", "F1", "accuracy"]
        ]

    plots.extend(_compute_confidence_histograms(test_case_name, metrics, confidence_range))
    plots.append(_compute_test_case_ovr_roc_curve(test_case_name, gt_labels, ground_truths, inferences))
    plots.append(_compute_test_case_confusion_matrix(test_case_name, ground_truths, metrics))
    plots = list(filter(lambda plot: plot is not None, plots))

    return plots


def _compute_test_case_confusion_matrix(
    test_case_name: str,
    ground_truths: List[GroundTruth],
    metrics: List[TestSampleMetrics],
) -> Optional[Plot]:
    gt_labels: Set[str] = set()
    pred_labels: Set[str] = set()
    none_label = "None"
    # actual to predicted to count
    confusion_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for gt, metric in zip(ground_truths, metrics):
        actual_label = gt.classification.label
        predicted_label = metric.label if metric.label is not None else none_label
        gt_labels.add(actual_label)
        pred_labels.add(predicted_label)
        confusion_matrix[actual_label][predicted_label] += 1

    if len(gt_labels) < 2:
        log.warn(f"skipping confusion matrix for {test_case_name}: single label test case")
        return None

    labels: Set[str] = {*gt_labels, *pred_labels}
    contains_none = none_label in labels
    sortable_labels = [label for label in labels if label != none_label]
    ordered_labels = sorted(sortable_labels) if not contains_none else [*sorted(sortable_labels), none_label]
    matrix = []
    for actual_label in ordered_labels:
        matrix.append([confusion_matrix[actual_label][predicted_label] for predicted_label in ordered_labels])
    return ConfusionMatrix(title="Label Confusion Matrix", labels=ordered_labels, matrix=matrix)


def _compute_test_case_ovr_roc_curve(
    test_case_name: str,
    labels: List[str],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> Optional[Plot]:
    if len(labels) > 10:
        log.warn(f"skipping one-vs-rest ROC curve for {test_case_name}: too many labels")
        return None

    curves = []
    for label in labels:
        y_true = [1 if gt.classification.label == label else 0 for gt in ground_truths]
        y_score = [get_label_confidence(label, inf.inferences) for inf in inferences]
        fpr_values, tpr_values = roc_curve(y_true, y_score)
        if len(fpr_values) > 0 and len(tpr_values) > 0:
            curves.append(Curve(x=fpr_values, y=tpr_values, label=label))

    if len(curves) > 0:
        return CurvePlot(
            title="Receiver Operating Characteristic (One-vs-Rest)",
            x_label="False Positive Rate (FPR)",
            y_label="True Positive Rate (TPR)",
            curves=curves,
        )
    return None


def _compute_test_case_metrics(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    metrics_test_samples: List[TestSampleMetrics],
    labels: List[str],
) -> TestCaseMetrics:
    classification_pairs = [
        (gt.classification.label, tsm.label) for gt, tsm in zip(ground_truths, metrics_test_samples)
    ]
    n_images = len(classification_pairs)
    class_level_metrics: List[ClassMetricsPerTestCase] = []
    for label in labels:
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
        macro_accuracy=np.average([class_metric.accuracy for class_metric in class_level_metrics]),
        macro_Precision=np.average([class_metric.Precision for class_metric in class_level_metrics]),
        macro_Recall=np.average([class_metric.Recall for class_metric in class_level_metrics]),
        macro_F1=np.average([class_metric.F1 for class_metric in class_level_metrics]),
        macro_FPR=np.average([class_metric.FPR for class_metric in class_level_metrics]),
    )


def evaluate_multiclass_classification(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration = dataclasses.field(default_factory=ThresholdConfiguration),
) -> EvaluationResults:
    metrics_test_sample: List[Tuple[TestSample, TestSampleMetrics]] = [
        (ts, _compute_test_sample_metric(configuration, gt, inf))
        for ts, gt, inf in zip(test_samples, ground_truths, inferences)
    ]
    test_sample_metrics: List[TestSampleMetrics] = [mts for _, mts in metrics_test_sample]
    confidence_scores = [mts.score for mts in test_sample_metrics if mts.label is not None]
    confidence_range = get_histogram_range(confidence_scores)

    metrics_test_case: List[Tuple[TestCase, TestCaseMetrics]] = []
    plots_test_case: List[Tuple[TestCase, List[Plot]]] = []
    for tc, tc_samples, tc_gts, tc_infs, tc_metrics in test_cases.iter(
        test_samples,
        ground_truths,
        inferences,
        test_sample_metrics,
    ):
        test_case_labels: Set[str] = set()
        gt_labels: Set[str] = set()
        for tc_gt in tc_gts:
            test_case_labels.add(tc_gt.classification.label)
            gt_labels.add(tc_gt.classification.label)
        for tc_inf in tc_infs:
            for inf in tc_inf.inferences:
                test_case_labels.add(inf.label)
        test_case_metrics = _compute_test_case_metrics(tc_samples, tc_gts, tc_metrics, sorted(gt_labels))
        test_case_plots = _compute_test_case_plots(
            tc.name,
            test_case_metrics.PerClass,
            sorted(test_case_labels),
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
        n_invalid=len([True for tsm in test_sample_metrics if tsm.label is not None]),
        n_correct=n_correct,
        overall_accuracy=n_correct / n_images if n_images > 0 else 0,
    )

    return EvaluationResults(
        metrics_test_sample=metrics_test_sample,
        metrics_test_case=metrics_test_case,
        plots_test_case=plots_test_case,
        metrics_test_suite=metrics_test_suite,
    )
