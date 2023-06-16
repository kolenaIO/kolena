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
from kolena.workflow.metrics import f1_score
from kolena.workflow.metrics import precision
from kolena.workflow.metrics import recall


def _compute_per_image_metrics(
    ground_truth: GroundTruth,
    inference: Inference,
    threshold_configuration: ThresholdConfiguration,
) -> PerImageMetrics:
    empty_metrics = PerImageMetrics(classification=None, margin=None, is_correct=False)
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
    per_class_metrics_by_class: Dict[str, PerClassMetrics],
    all_classes: List[str],
    display_name: Optional[str] = None,
) -> Optional[BarPlot]:
    title = f"{display_name or metric_name} vs. Class"
    values = [getattr(per_class_metrics_by_class[label], metric_name) for label in all_classes]
    valid_pairs = [(label, value) for label, value in zip(all_classes, values) if value != 0.0]

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
    per_image_metrics: List[PerImageMetrics],
    confidence_range: Optional[Tuple[float, float, int]],
) -> List[Histogram]:
    if confidence_range is None:
        log.warn("skipping confidence histograms for test case: unsupported confidence range")
        return []

    confidence_all = [mts.classification.score for mts in per_image_metrics if mts.classification is not None]
    confidence_correct = [
        mts.classification.score for mts in per_image_metrics if mts.classification is not None and mts.is_correct
    ]
    confidence_incorrect = [
        mts.classification.score for mts in per_image_metrics if mts.classification is not None and not mts.is_correct
    ]

    plots = [
        _as_confidence_histogram("Score Distribution (All)", confidence_all, confidence_range),
        _as_confidence_histogram("Score Distribution (Correct)", confidence_correct, confidence_range),
        _as_confidence_histogram("Score Distribution (Incorrect)", confidence_incorrect, confidence_range),
    ]
    return plots


def _compute_test_case_plots(
    all_classes: List[str],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    per_image_metrics: List[PerImageMetrics],
    per_class_metrics_by_class: Dict[str, PerClassMetrics],
    confidence_range: Optional[Tuple[float, float, int]],
) -> List[Plot]:
    gt_labels = {gt.classification.label for gt in ground_truths}
    plots: List[Optional[Plot]] = [_as_class_metric_plot("FPR", per_class_metrics_by_class, all_classes)]

    if len(gt_labels) > 2:  # only plot Precision, Recall, F1 vs. Class when there are multiple classes in the test case
        plots.append(_as_class_metric_plot("Precision", per_class_metrics_by_class, all_classes))
        plots.append(
            _as_class_metric_plot("Recall", per_class_metrics_by_class, all_classes, display_name="Recall (TPR)"),
        )
        plots.append(_as_class_metric_plot("F1", per_class_metrics_by_class, all_classes))

    plots.extend(_compute_confidence_histograms(per_image_metrics, confidence_range))
    plots.append(_compute_test_case_ovr_roc_curve(list(gt_labels), ground_truths, inferences))
    plots.append(_compute_test_case_confusion_matrix(ground_truths, per_image_metrics))

    return [plot for plot in plots if plot is not None]


def _compute_test_case_confusion_matrix(
    ground_truths: List[GroundTruth],
    per_image_metrics: List[PerImageMetrics],
) -> Optional[Plot]:
    gt_labels: Set[str] = set()
    pred_labels: Set[Optional[str]] = set()
    # actual to predicted to count
    confusion_matrix: Dict[str, Dict[Optional[str], int]] = defaultdict(lambda: defaultdict(int))
    for gt, metric in zip(ground_truths, per_image_metrics):
        actual_label = gt.classification.label
        predicted_label = metric.classification.label  # note: includes None
        gt_labels.add(actual_label)
        pred_labels.add(predicted_label)
        confusion_matrix[actual_label][predicted_label] += 1

    if len(gt_labels) < 2:
        log.info("skipping confusion matrix for single label test case")
        return None

    labels: Set[str] = {*gt_labels, *pred_labels}
    sortable_labels = [label for label in labels if label is not None]
    ordered_labels = sorted(sortable_labels) if None not in labels else [*sorted(sortable_labels), str(None)]
    matrix: List[List[Optional[int]]] = [
        [confusion_matrix[actual_label][predicted_label] for predicted_label in ordered_labels]
        for actual_label in ordered_labels
    ]
    return ConfusionMatrix(title="Label Confusion Matrix", labels=ordered_labels, matrix=matrix)


def _compute_test_case_ovr_roc_curve(
    all_classes: List[str],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> Optional[Plot]:
    n_max = 10
    if len(all_classes) > n_max:
        log.warn(f"skipping one-vs-rest ROC curve for test case: too many labels (got {len(all_classes)}, max {n_max})")
        return None

    curves = []
    for label in all_classes:
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
    all_classes: List[str],
    ground_truths: List[GroundTruth],
    per_image_metrics: List[PerImageMetrics],
) -> Dict[str, PerClassMetrics]:
    per_class_metrics_by_label = {}
    for base_label in all_classes:
        n_tp = 0
        n_fp = 0
        n_fn = 0
        n_tn = 0
        for gt, mts in zip(ground_truths, per_image_metrics):
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
        per_class_metrics_by_label[base_label] = PerClassMetrics(
            label=base_label,
            Precision=precision(n_tp, n_fp),
            Recall=recall(n_tp, n_fn),
            F1=f1_score(n_tp, n_fp, n_fn),
            FPR=n_fp / (n_fp + n_tn) if n_fp + n_tn > 0 else 0,
        )
    return per_class_metrics_by_label


def _compute_aggregate_metrics(
    ground_truths: List[GroundTruth],
    per_image_metrics: List[PerImageMetrics],
    per_class_metrics_by_class: Dict[str, PerClassMetrics],
) -> AggregateMetrics:
    n_correct = len([metric for metric in per_image_metrics if metric.is_correct])
    n_total = len(ground_truths)
    test_case_classes = {gt.classification.label for gt in ground_truths}

    def macro_metric(metric_name: str) -> float:
        metrics = [getattr(per_class_metrics_by_class[class_name], metric_name) for class_name in test_case_classes]
        return sum(metrics) / len(metrics)

    return AggregateMetrics(
        n_correct=n_correct,
        n_incorrect=n_total - n_correct,
        Accuracy=n_correct / n_total,
        Precision_macro=macro_metric("Precision"),
        Recall_macro=macro_metric("Recall"),
        F1_macro=macro_metric("F1"),
        FPR_macro=macro_metric("FPR"),
        PerClass=sorted(list(per_class_metrics_by_class.values()), key=lambda pcm: pcm.label),
    )


def _compute_test_suite_metrics(
    per_image_metrics: List[PerImageMetrics],
    aggregate_metrics: List[AggregateMetrics],
) -> TestSuiteMetrics:
    def _compute_variance(metric_name: str) -> float:
        if len(aggregate_metrics) == 0:
            return 0
        return float(np.var([getattr(metrics, metric_name) for metrics in aggregate_metrics]))

    return TestSuiteMetrics(
        n_images=len(per_image_metrics),
        n_images_skipped=len([mts for mts in per_image_metrics if mts.classification is None]),
        variance_Accuracy=_compute_variance("Accuracy"),
        variance_Precision_macro=_compute_variance("Precision_macro"),
        variance_Recall_macro=_compute_variance("Recall_macro"),
        variance_F1_macro=_compute_variance("F1_macro"),
        variance_FPR_macro=_compute_variance("FPR_macro"),
    )


def evaluate_multiclass_classification(
    images: List[Image],
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
    all_labels_set = {
        *(gt.classification.label for gt in ground_truths),
        *(label.label for inf in inferences for label in inf.inferences),
    }
    all_labels = sorted(all_labels_set)

    per_image_metrics = [
        _compute_per_image_metrics(gt, inf, configuration) for gt, inf in zip(ground_truths, inferences)
    ]
    confidence_scores = [pim.classification.score for pim in per_image_metrics if pim.classification is not None]
    confidence_range = get_histogram_range(confidence_scores)

    metrics_test_case: List[Tuple[TestCase, AggregateMetrics]] = []
    plots_test_case: List[Tuple[TestCase, List[Plot]]] = []
    for tc, *tc_samples in test_cases.iter(images, ground_truths, inferences, per_image_metrics):
        _, tc_gts, tc_infs, tc_metrics = tc_samples

        per_class_metrics = _compute_per_class_metrics(all_labels, tc_gts, tc_metrics)
        aggregate_metrics = _compute_aggregate_metrics(tc_gts, tc_metrics, per_class_metrics)
        metrics_test_case.append((tc, aggregate_metrics))

        test_case_plots = _compute_test_case_plots(
            all_labels,
            tc_gts,
            tc_infs,
            tc_metrics,
            per_class_metrics,
            confidence_range,
        )
        plots_test_case.append((tc, test_case_plots))

    metrics_test_suite = _compute_test_suite_metrics(per_image_metrics, [metric for _, metric in metrics_test_case])

    return EvaluationResults(
        metrics_test_sample=list(zip(images, per_image_metrics)),
        metrics_test_case=metrics_test_case,
        plots_test_case=plots_test_case,
        metrics_test_suite=metrics_test_suite,
    )
