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

from kolena._utils import log
from kolena.classification.multiclass._utils import get_histogram_range
from kolena.classification.multiclass._utils import get_label_confidence
from kolena.classification.multiclass._utils import roc_curve
from kolena.classification.multiclass.workflow import AggregateMetrics
from kolena.classification.multiclass.workflow import GroundTruth
from kolena.classification.multiclass.workflow import Inference
from kolena.classification.multiclass.workflow import PerClassMetrics
from kolena.classification.multiclass.workflow import PerImageMetrics
from kolena.classification.multiclass.workflow import TestCase
from kolena.classification.multiclass.workflow import TestSuiteMetrics
from kolena.classification.multiclass.workflow import ThresholdConfiguration
from kolena.workflow import BarPlot
from kolena.workflow import ConfusionMatrix
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import EvaluationResults
from kolena.workflow import Histogram
from kolena.workflow import Image
from kolena.workflow import Plot
from kolena.workflow import TestCases


def _compute_per_image_metrics(
    threshold_configuration: ThresholdConfiguration,
    ground_truth: GroundTruth,
    inference: Inference,
) -> PerImageMetrics:
    empty_metrics = PerImageMetrics(
        classification=None,
        margin=None,
        is_correct=False,
    )
    if len(inference.inferences) == 0:
        return empty_metrics

    sorted_indices = np.argsort([label.score for label in inference.inferences])
    match = inference.inferences[sorted_indices[-1]]
    predicted_label, confidence_score = match.label, match.score
    margin = confidence_score - inference.inferences[sorted_indices[-2]].score if len(sorted_indices) > 1 else None

    if threshold_configuration.threshold is not None and confidence_score < threshold_configuration.threshold:
        return empty_metrics

    is_correct = predicted_label == ground_truth.classification.label
    return PerImageMetrics(
        classification=match,
        margin=margin,
        is_correct=is_correct,
    )


def _as_class_metric_plot(
    metric_name: str,
    metrics_by_label: Dict[str, PerClassMetrics],
    labels: List[str],
    display_name: Optional[str] = None,
) -> Optional[BarPlot]:
    title = f"{display_name or metric_name} vs. Class"
    values = [getattr(metrics_by_label[label], metric_name) for label in labels]
    valid_pairs = [(label, value) for label, value in zip(labels, values) if value != 0.0]

    if len(valid_pairs) == 0:
        return None

    return BarPlot(
        title=title,
        x_label="Class",
        y_label=metric_name,
        labels=[label for label, _ in valid_pairs],
        values=[value for _, value in valid_pairs],
    )


def _as_confidence_histogram(
    title: str,
    confidence_scores: List[float],
    confidence_range: Tuple[float, float, int] = (0, 1, 25),
) -> Histogram:
    min_range, max_range, bins = confidence_range
    hist, bins = np.histogram(confidence_scores, bins=bins, range=(min_range, max_range))
    return Histogram(title=title, x_label="Confidence", y_label="Count", buckets=list(bins), frequency=list(hist))


def _compute_confidence_histograms(
    test_case_name: str,
    metrics: List[PerImageMetrics],
    confidence_range: Optional[Tuple[float, float, int]],
) -> List[Histogram]:
    if confidence_range is None:
        log.warn(f"skipping confidence histograms for {test_case_name}: unsupported confidence range")
        return []

    confidence_all = [mts.classification.score for mts in metrics if mts.classification is not None]
    confidence_correct = [
        mts.classification.score for mts in metrics if mts.classification is not None and mts.is_correct
    ]
    confidence_incorrect = [
        mts.classification.score for mts in metrics if mts.classification is not None and not mts.is_correct
    ]

    plots = [
        _as_confidence_histogram("Score Distribution (All)", confidence_all, confidence_range),
        _as_confidence_histogram("Score Distribution (Correct)", confidence_correct, confidence_range),
        _as_confidence_histogram("Score Distribution (Incorrect)", confidence_incorrect, confidence_range),
    ]
    return plots


def _compute_test_case_plots(
    test_case_name: str,
    labels: List[str],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics: List[PerImageMetrics],
    metrics_by_label: Dict[str, PerClassMetrics],
    confidence_range: Optional[Tuple[float, float, int]],
) -> List[Plot]:
    gt_labels = {gt.classification.label for gt in ground_truths}
    plots: List[Optional[Plot]] = [_as_class_metric_plot("FPR", metrics_by_label, labels)]

    if len(gt_labels) > 2:  # only plot Precision, Recall, F1 vs. Class when there are multiple classes in the test case
        plots.append(_as_class_metric_plot("Precision", metrics_by_label, labels))
        plots.append(_as_class_metric_plot("Recall", metrics_by_label, labels, display_name="Recall (TPR)"))
        plots.append(_as_class_metric_plot("F1", metrics_by_label, labels))

    plots.extend(_compute_confidence_histograms(test_case_name, metrics, confidence_range))
    plots.append(_compute_test_case_ovr_roc_curve(test_case_name, list(gt_labels), ground_truths, inferences))
    plots.append(_compute_test_case_confusion_matrix(test_case_name, ground_truths, metrics))

    return [plot for plot in plots if plot is not None]


def _compute_test_case_confusion_matrix(
    test_case_name: str,
    ground_truths: List[GroundTruth],
    metrics: List[PerImageMetrics],
) -> Optional[Plot]:
    gt_labels: Set[str] = set()
    pred_labels: Set[str] = set()
    none_label = "None"
    # actual to predicted to count
    confusion_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for gt, metric in zip(ground_truths, metrics):
        actual_label = gt.classification.label
        predicted_label = metric.classification.label if metric.classification is not None else none_label
        gt_labels.add(actual_label)
        pred_labels.add(predicted_label)
        confusion_matrix[actual_label][predicted_label] += 1

    if len(gt_labels) < 2:
        log.info(f"skipping confusion matrix for {test_case_name}: single label test case")
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

    if len(curves) == 0:
        return None

    return CurvePlot(
        title="Receiver Operating Characteristic (One-vs-Rest)",
        x_label="False Positive Rate (FPR)",
        y_label="True Positive Rate (TPR)",
        curves=curves,
    )


def _compute_per_class_metrics(
    labels: List[str],
    test_samples: List[Image],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics_test_samples: List[PerImageMetrics],
) -> Dict[str, PerClassMetrics]:
    per_class_metrics_by_label = {}
    for base_label in labels:
        n_tp = 0
        n_fp = 0
        n_fn = 0
        n_tn = 0
        for ts, gt, inf, mts in zip(test_samples, ground_truths, inferences, metrics_test_samples):
            gt_label = gt.classification.label
            predicted_label = mts.classification.label if mts.classification is not None else None
            if gt_label == base_label and predicted_label == base_label:
                n_tp += 1
            if gt_label == base_label and predicted_label != base_label:
                n_fn += 1
            if gt_label != base_label and predicted_label == base_label:
                n_fp += 1
            if gt_label != base_label and predicted_label != base_label:
                n_tn += 1
        precision = n_tp / (n_tp + n_fp) if n_tp + n_fp > 0 else 0
        recall = n_tp / (n_tp + n_fn) if n_tp + n_fn > 0 else 0
        fpr = n_fp / (n_fp + n_tn) if n_fp + n_tn > 0 else 0
        f1_score = (2 * precision * recall) / (precision + recall) if precision + recall > 0 else 0
        per_class_metrics_by_label[base_label] = PerClassMetrics(
            label=base_label,
            F1=f1_score,
            Precision=precision,
            Recall=recall,
            FPR=fpr,
        )
    return per_class_metrics_by_label


def _compute_aggregate_metrics(
    test_samples: List[Image],
    ground_truths: List[GroundTruth],
    metrics_test_samples: List[PerImageMetrics],
    metrics_by_label: Dict[str, PerClassMetrics],
) -> AggregateMetrics:
    n_correct = len([metric for metric in metrics_test_samples if metric.is_correct])
    n_images = len(test_samples)
    labels = {gt.classification.label for gt in ground_truths}

    macro_metrics_by_name: Dict[str, float] = {}
    non_metric_field_names = {"label"}
    for field in dataclasses.fields(PerClassMetrics):
        if field.name in non_metric_field_names:
            continue
        # only consider labels that exist within this test case
        metrics = [getattr(metrics_by_label[label], field.name) for label in labels]
        macro_metrics_by_name[field.name] = sum(metrics) / len(metrics)

    return AggregateMetrics(
        n_correct=n_correct,
        n_incorrect=n_images - n_correct,
        Accuracy=n_correct / n_images,
        Precision_macro=macro_metrics_by_name["Precision"],
        Recall_macro=macro_metrics_by_name["Recall"],
        F1_macro=macro_metrics_by_name["F1"],
        FPR_macro=macro_metrics_by_name["FPR"],
    )


def _compute_test_suite_metrics(
    test_sample_metrics: List[PerImageMetrics],
    test_case_metrics: List[AggregateMetrics],
) -> TestSuiteMetrics:
    def _compute_variance(metric_name: str) -> float:
        return float(np.var([getattr(metrics, metric_name) for metrics in test_case_metrics]) / len(test_case_metrics))

    return TestSuiteMetrics(
        n_images=len(test_sample_metrics),
        n_images_skipped=len([mts for mts in test_sample_metrics if mts.classification is None]),
        variance_Accuracy=_compute_variance("Accuracy"),
        variance_Precision_macro=_compute_variance("Precision_macro"),
        variance_Recall_macro=_compute_variance("Recall_macro"),
        variance_F1_macro=_compute_variance("F1_macro"),
        variance_FPR_macro=_compute_variance("FPR_macro"),
    )


def evaluate_multiclass_classification(
    test_samples: List[Image],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration,
) -> EvaluationResults:
    """
    [Function-based evaluator implementation][kolena.workflow.BasicEvaluatorFunction] for the pre-built Multiclass
    Classification workflow.

    It is not necessary to use this definition directly when testing with
    [`kolena.classification.multiclass.test`][kolena.classification.multiclass.test], which is already bound to this
    evaluator implementation. Provide this definition when testing with [`kolena.workflow.test`][kolena.workflow.test]
    or [`kolena.workflow.TestRun`][kolena.workflow.TestRun].
    """
    labels_set: Set[str] = set()
    for gt in ground_truths:
        labels_set.add(gt.classification.label)
    for inf in inferences:
        for label in inf.inferences:
            labels_set.add(label.label)
    labels = sorted(labels_set)

    metrics_test_sample: List[Tuple[Image, PerImageMetrics]] = [
        (ts, _compute_per_image_metrics(configuration, gt, inf))
        for ts, gt, inf in zip(test_samples, ground_truths, inferences)
    ]
    test_sample_metrics: List[PerImageMetrics] = [mts for _, mts in metrics_test_sample]
    confidence_scores = [mts.classification.score for mts in test_sample_metrics if mts.classification is not None]
    confidence_range = get_histogram_range(confidence_scores)

    metrics_test_case: List[Tuple[TestCase, AggregateMetrics]] = []
    plots_test_case: List[Tuple[TestCase, List[Plot]]] = []
    for tc, tc_samples, tc_gts, tc_infs, tc_metrics in test_cases.iter(
        test_samples,
        ground_truths,
        inferences,
        test_sample_metrics,
    ):
        per_class_metrics = _compute_per_class_metrics(labels, tc_samples, tc_gts, tc_infs, tc_metrics)
        aggregate_metrics = _compute_aggregate_metrics(tc_samples, tc_gts, tc_metrics, per_class_metrics)
        metrics_test_case.append((tc, aggregate_metrics))
        test_case_plots = _compute_test_case_plots(
            tc.name,
            labels,
            tc_gts,
            tc_infs,
            tc_metrics,
            per_class_metrics,
            confidence_range,
        )
        plots_test_case.append((tc, test_case_plots))

    all_test_case_metrics = [metric for _, metric in metrics_test_case]
    metrics_test_suite = _compute_test_suite_metrics(test_sample_metrics, all_test_case_metrics)

    return EvaluationResults(
        metrics_test_sample=metrics_test_sample,
        metrics_test_case=metrics_test_case,
        plots_test_case=plots_test_case,
        metrics_test_suite=metrics_test_suite,
    )
