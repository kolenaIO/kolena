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

from kolena._experimental.classification.workflow import ClassMetricsPerTestCase
from kolena._experimental.classification.workflow import GroundTruth
from kolena._experimental.classification.workflow import Inference
from kolena._experimental.classification.workflow import TestSampleMetrics
from kolena._extras.metrics import sklearn_metrics
from kolena._utils import log
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.plot import BarPlot
from kolena.workflow.plot import ConfusionMatrix
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot
from kolena.workflow.plot import Histogram
from kolena.workflow.plot import Plot


def get_label_confidence(label: str, inference_labels: List[ScoredClassificationLabel]) -> float:
    return next((inf.score for inf in inference_labels if inf.label == label), 0)


def get_histogram_range(values: List[float]) -> Optional[Tuple[float, float, int]]:
    if len(values) == 0:
        log.warn("insufficient values provided for confidence histograms")
        return None

    NUM002 = 0.02
    lower = min(values)
    higher = max(values)
    if lower < 0.0 or higher > 1.0:
        log.warn(
            f"values out of range for confidence histograms: expecting [0, 1], got [{lower:.3f}, {higher:.3f}]",
        )
        return None

    # round to 0.02
    min_score = (lower + 1e-9) // NUM002 * NUM002
    max_score = (higher - 1e-9) // NUM002 * NUM002 + NUM002

    if max_score == min_score:
        if max_score < 0.5:
            max_score = min_score + NUM002
        else:
            min_score = max_score - NUM002

    return min_score, max_score, (max_score - min_score - 1e-9) // NUM002 + 1


def create_histogram(
    values: List[float],
    range: Tuple[float, float, int],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> Histogram:
    min_range, max_range, bins = range
    frequency, buckets = np.histogram(
        values,
        bins=bins,
        range=(min_range, max_range),
    )
    return Histogram(
        title=title,
        x_label=x_label,
        y_label=y_label,
        buckets=list(buckets),
        frequency=list(frequency),
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


def metric_bar_plot_by_class(
    metric_name: str,
    per_class_metrics: List[ClassMetricsPerTestCase],
) -> Optional[BarPlot]:
    valid_metric_names = {"n_correct", "n_incorrect", "Accuracy", "Precision", "Recall", "F1", "FPR"}
    if metric_name not in valid_metric_names:
        return None
    values = [(pcm.label, getattr(pcm, metric_name)) for pcm in per_class_metrics]
    valid_pairs = [(label, value) for label, value in values if value != 0.0]
    return (
        BarPlot(
            title=f"{metric_name} by Class",
            x_label="Class",
            y_label=metric_name,
            labels=[label for label, _ in valid_pairs],
            values=[value for _, value in valid_pairs],
        )
        if len(valid_pairs) > 0
        else None
    )


# assuming multiclass, where no classifications are None
def compute_test_case_roc_curves(
    labels: List[str],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
) -> Optional[Plot]:
    curves: List[Curve] = []
    for label in sorted(labels):
        y_true = [1 if gt.classification.label == label else 0 for gt in ground_truths]
        y_score = [get_label_confidence(label, inf.inferences) for inf in inferences]
        fpr_values, tpr_values, _ = sklearn_metrics.roc_curve(y_true=y_true, y_score=y_score)

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


# handles both binary and multi, assumes binary contains None type
def compute_test_case_confusion_matrix(
    ground_truths: List[GroundTruth],
    metrics: List[TestSampleMetrics],
) -> Optional[Plot]:
    if len(ground_truths) != len(metrics):
        log.warn(
            f"ground_truths ({len(ground_truths)}) and metrics ({len(metrics)}) "
            "differ in length for a confusion matrix",
        )
        return None

    if len(ground_truths) == 0:
        log.warn("no ground_truths provided for a confusion matrix")
        return None

    # if there are only None classifications in the ground_truths, this leads to a bad confusion matrix
    if not any(gt.classification for gt in ground_truths):
        log.warn("no positive class in ground_truths provided for a confusion matrix")
        return None

    labels: Set[str] = set()
    none_label = "None"
    confusion_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for gt, metric in zip(ground_truths, metrics):
        actual_label = gt.classification.label if gt.classification else none_label
        predicted_label = metric.classification.label if metric.classification else none_label
        labels.add(actual_label)
        labels.add(predicted_label)
        confusion_matrix[actual_label][predicted_label] += 1

    ordered_labels = sorted([label for label in labels if label != none_label])

    # create a 2 by 2 confusion matrix to outline TP, FP, FN, and TN for binary classification
    if len(ordered_labels) == 1:
        label = ordered_labels[0]
        matrix = [
            [confusion_matrix[label][label], confusion_matrix[label][none_label]],
            [confusion_matrix[none_label][label], confusion_matrix[none_label][none_label]],
        ]
        return ConfusionMatrix(title="Label Confusion Matrix", labels=[label, f"Not {label}"], matrix=matrix)

    if none_label in labels:
        ordered_labels.append(none_label)

    matrix = []
    for actual_label in ordered_labels:
        matrix.append([confusion_matrix[actual_label][predicted_label] for predicted_label in ordered_labels])
    return ConfusionMatrix(title="Label Confusion Matrix", labels=ordered_labels, matrix=matrix)
