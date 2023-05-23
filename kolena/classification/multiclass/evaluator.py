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
from dataclasses import make_dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np

from kolena._utils import log
from kolena.classification.multiclass._utils import get_histogram_range
from kolena.classification.multiclass._utils import get_label_confidence
from kolena.classification.multiclass._utils import roc_curve
from kolena.classification.multiclass.workflow import AggregatedMetrics
from kolena.classification.multiclass.workflow import GroundTruth
from kolena.classification.multiclass.workflow import Inference
from kolena.classification.multiclass.workflow import InferenceLabel
from kolena.classification.multiclass.workflow import TestCase
from kolena.classification.multiclass.workflow import TestCaseMetrics
from kolena.classification.multiclass.workflow import TestSample
from kolena.classification.multiclass.workflow import TestSampleMetrics
from kolena.classification.multiclass.workflow import TestSuiteMetrics
from kolena.classification.multiclass.workflow import ThresholdConfiguration
from kolena.workflow import BarPlot
from kolena.workflow import ConfusionMatrix
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import EvaluationResults
from kolena.workflow import Histogram
from kolena.workflow import Plot
from kolena.workflow import TestCases
from kolena.workflow.annotation import ScoredClassificationLabel

Result = Tuple[TestSample, GroundTruth, Inference]


def _compute_test_sample_metric(
    threshold_configuration: ThresholdConfiguration,
    ground_truth: GroundTruth,
    inference: Inference,
) -> TestSampleMetrics:
    empty_metrics = TestSampleMetrics(
        classification=None,
        margin=None,
        is_correct=False,
    )
    if len(inference.inferences) == 0:
        return empty_metrics

    sorted_indices = np.argsort([label.score for label in inference.inferences])
    match = inference.inferences[sorted_indices[-1]]
    predicted_label, confidence_score = match.label, match.score
    margin: Optional[float] = None
    if len(sorted_indices) > 1:
        second_closest: Union[ScoredClassificationLabel, InferenceLabel] = inference.inferences[sorted_indices[-2]]
        margin = confidence_score - second_closest.score

    if threshold_configuration.threshold is not None and confidence_score < threshold_configuration.threshold:
        return empty_metrics

    is_correct = predicted_label == ground_truth.classification.label
    return TestSampleMetrics(
        classification=match,
        margin=margin,
        is_correct=is_correct,
    )


def _as_class_metric_plot(
    metric_name: str,
    metrics_by_label: Dict[str, AggregatedMetrics],
    labels: List[str],
) -> Optional[BarPlot]:
    if metric_name == "Recall":
        title = f"{metric_name} (TPR) vs. Class"
    else:
        title = f"{metric_name} vs. Class"

    values = [getattr(metrics_by_label[label], metric_name) for label in labels]
    valid_pairs = [(label, value) for label, value in zip(labels, values) if value != 0.0]
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
    metrics: List[TestSampleMetrics],
    metrics_by_label: Dict[str, AggregatedMetrics],
    confidence_range: Optional[Tuple[float, float, int]],
) -> List[Plot]:
    gt_labels = {gt.classification.label for gt in ground_truths}
    plots: List[Plot] = [
        _as_class_metric_plot(field.name, metrics_by_label, labels)
        for field in dataclasses.fields(AggregatedMetrics)
        if len(gt_labels) > 2
        or field.name not in ["Precision", "Recall"]  # Omit single-class TC from precision and recall plots
    ]

    plots.extend(_compute_confidence_histograms(test_case_name, metrics, confidence_range))
    plots.append(_compute_test_case_ovr_roc_curve(test_case_name, labels, ground_truths, inferences))
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
        predicted_label = metric.classification.label if metric.classification is not None else none_label
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
    curves = []
    for label in labels:
        y_true = [1 if gt.classification.label == label else 0 for gt in ground_truths]
        y_score = [get_label_confidence(label, inf.inferences) for inf in inferences]
        fpr_values, tpr_values = roc_curve(y_true, y_score)
        if len(fpr_values) > 0 and len(tpr_values) > 0:
            curves.append(Curve(x=fpr_values, y=tpr_values, label=label))

    if len(curves) > 0:
        if len(curves) > 10:
            log.warn(f"skipping one-vs-rest ROC curve for {test_case_name}: too many labels")
            return None

        return CurvePlot(
            title="Receiver Operating Characteristic (One-vs-Rest)",
            x_label="False Positive Rate (FPR)",
            y_label="True Positive Rate (TPR)",
            curves=curves,
        )
    return None


def _aggregate_label_metrics(
    labels: List[str],
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    metrics_test_samples: List[TestSampleMetrics],
) -> Dict[str, AggregatedMetrics]:
    aggregated_metrics = {}
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
        aggregated_metrics[base_label] = AggregatedMetrics(
            F1=f1_score,
            Precision=precision,
            Recall=recall,
            FPR=fpr,
        )
    return aggregated_metrics


def _compute_test_case_metrics(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    metrics_test_samples: List[TestSampleMetrics],
    metrics_by_label: Dict[str, AggregatedMetrics],
) -> TestCaseMetrics:
    n_correct = len([metric for metric in metrics_test_samples if metric.is_correct])
    n_images = len(test_samples)
    n_incorrect = n_images - n_correct
    accuracy = n_correct / n_images
    labels = {gt.classification.label for gt in ground_truths}

    macro_metrics_by_name: Dict[str, float] = {}
    for field in dataclasses.fields(AggregatedMetrics):
        metric_name = field.name
        metrics = [getattr(metrics_by_label[label], metric_name) for label in labels]
        macro_metrics_by_name[metric_name] = sum(metrics) / len(metrics)

    return TestCaseMetrics(
        n_correct=n_correct,
        n_incorrect=n_incorrect,
        accuracy=accuracy,
        macro_precision=macro_metrics_by_name["Precision"],
        macro_recall=macro_metrics_by_name["Recall"],
        macro_f1=macro_metrics_by_name["F1"],
        macro_tpr=macro_metrics_by_name["Recall"],
        macro_fpr=macro_metrics_by_name["FPR"],
    )


def _compute_test_suite_metrics(
    labels: List[str],
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_sample_metrics: List[TestSampleMetrics],
    test_case_metrics: List[TestCaseMetrics],
) -> TestSuiteMetrics:
    n_images = len(test_sample_metrics)
    n_correct = len([metric for metric in test_sample_metrics if metric.is_correct])
    mean_test_case_accuracy = sum([metric.accuracy for metric in test_case_metrics]) / len(test_case_metrics)

    values: Dict[str, Any] = dict(
        n_images=n_images,
        n_correct=n_correct,
        mean_test_case_accuracy=mean_test_case_accuracy,
    )
    fields: Dict[str, Type] = {}

    metrics_by_label = _aggregate_label_metrics(labels, test_samples, ground_truths, inferences, test_sample_metrics)
    for field in dataclasses.fields(AggregatedMetrics):
        attr = field.name
        label_values = [getattr(metric, attr) for metric in metrics_by_label.values()]
        mean_field_name = f"mean_{attr}"
        var_field_name = f"variance_{attr}"
        fields[mean_field_name] = float
        fields[var_field_name] = float
        values[mean_field_name] = np.mean(label_values)
        values[var_field_name] = np.var(label_values)

    dc = make_dataclass(
        "WorkflowTestSuiteMetrics",
        bases=(TestSuiteMetrics,),
        fields=list(fields.items()),
        frozen=True,
    )
    return dc(**values)


def MulticlassClassificationEvaluator(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    configuration: ThresholdConfiguration = dataclasses.field(default_factory=ThresholdConfiguration),
) -> EvaluationResults:
    labels_set: Set[str] = set()
    for gt in ground_truths:
        labels_set.add(gt.classification.label)
    for inf in inferences:
        for label in inf.inferences:
            labels_set.add(label.label)
    labels = sorted(labels_set)

    metrics_test_sample: List[Tuple[TestSample, TestSampleMetrics]] = [
        (ts, _compute_test_sample_metric(configuration, gt, inf))
        for ts, gt, inf in zip(test_samples, ground_truths, inferences)
    ]
    test_sample_metrics: List[TestSampleMetrics] = [mts for _, mts in metrics_test_sample]
    confidence_scores = [mts.classification.score for mts in test_sample_metrics if mts.classification is not None]
    confidence_range = get_histogram_range(confidence_scores)

    metrics_test_case: List[Tuple[TestCase, TestCaseMetrics]] = []
    plots_test_case: List[Tuple[TestCase, List[Plot]]] = []
    for tc, tc_samples, tc_gts, tc_infs, tc_metrics in test_cases.iter(
        test_samples,
        ground_truths,
        inferences,
        test_sample_metrics,
    ):
        aggregated_label_metrics = _aggregate_label_metrics(labels, tc_samples, tc_gts, tc_infs, tc_metrics)
        test_case_metrics = _compute_test_case_metrics(tc_samples, tc_gts, tc_metrics, aggregated_label_metrics)
        metrics_test_case.append((tc, test_case_metrics))
        test_case_plots = _compute_test_case_plots(
            tc.name,
            labels,
            tc_gts,
            tc_infs,
            tc_metrics,
            aggregated_label_metrics,
            confidence_range,
        )
        plots_test_case.append((tc, test_case_plots))

    all_test_case_metrics = [metric for _, metric in metrics_test_case]
    metrics_test_suite = _compute_test_suite_metrics(
        labels,
        test_samples,
        ground_truths,
        inferences,
        test_sample_metrics,
        all_test_case_metrics,
    )

    return EvaluationResults(
        metrics_test_sample=metrics_test_sample,
        metrics_test_case=metrics_test_case,
        plots_test_case=plots_test_case,
        metrics_test_suite=metrics_test_suite,
    )
