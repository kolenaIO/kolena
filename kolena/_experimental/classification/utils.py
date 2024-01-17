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
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np

from kolena._utils import log
from kolena.workflow.annotation import Label
from kolena.workflow.annotation import ScoredLabel
from kolena.workflow.metrics import f1_score
from kolena.workflow.metrics import precision
from kolena.workflow.metrics import recall
from kolena.workflow.plot import ConfusionMatrix
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot
from kolena.workflow.plot import Histogram


def get_label_confidence(label: str, inference_labels: List[ScoredLabel]) -> float:
    """
    Returns the confidence score of the specified `label` from a list of confidence scores for each label.

    :param label: The label whose confidence score to return.
    :param inference_labels: The list of confidence scores for each label. For `N`-class problem, expected to have
        `N` entries, one for each class.
    :return: The confidence score of the specified `label`. If the `label` doesn't exist in `inference_labels`
        then returns 0.
    """
    return next((inf.score for inf in inference_labels if inf.label == label), 0)


def get_histogram_range(values: List[float]) -> Optional[Tuple[float, float, int]]:
    """
    Computes an ideal range for a confidence score histograms given a list of confidence scores.

    :param values: The list of confidence scores, [0, 1].
    :return: A tuple of min, max and # of bins for a confidence score histograms. The range is rounded up/down to
        the nearest 0.02. The bin size is 0.02.
    """
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

    return min_score, max_score, int((max_score - min_score - 1e-9) // NUM002) + 1


def create_histogram(
    values: List[float],
    range: Tuple[float, float, int],
    title: str = "",
    x_label: str = "",
    y_label: str = "",
) -> Histogram:
    """
    Creates a [`Histogram`][kolena.workflow.plot.Histogram] for the specified range and the number of bins.

    :param values: The list of confidence scores to plot.
    :param range: The min, max and # of bins of the histogram.
    :param title: The title of the plot.
    :param x_label: The label on the x-axis.
    :param y_label: The label on the y-axis.
    :return: The [`Histogram`][kolena.workflow.plot.Histogram].
    """
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


def compute_confusion_matrix(
    ground_truths: List[str],
    inferences: List[str],
    title: str = "Confusion Matrix",
    labels: Optional[List[str]] = None,
) -> Optional[ConfusionMatrix]:
    """
    Computes confusion matrix given a list of ground truth and inference labels.

    For a binary classification case, a 2x2 confusion matrix with the count of TP, FP, FN, and TP is computed.

    :param ground_truths: The ground truth labels.
    :param inferences: The inference labels.
    :param title: The title of confusion matrix.
    :param labels: The list of labels to index the matrix. This may be used to reorder or select a subset of labels.
        By default, labels that appear at least once in `ground_truths` or `inferences` are used in sorted order.
    :return: The [`ConfusionMatrix`][kolena.workflow.plot.ConfusionMatrix].

    """
    if len(ground_truths) != len(inferences):
        log.warn(
            f"ground truth labels ({len(ground_truths)}) and inference labels ({len(inferences)}) "
            "differ in length for a confusion matrix — expecting equal length",
        )
        return None

    if len(ground_truths) == 0:
        log.warn("no ground_truths provided for a confusion matrix")
        return None

    confusion_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for gt, inf in zip(ground_truths, inferences):
        confusion_matrix[gt][inf] += 1

    if labels is None:
        labels = list({label for label in ground_truths} | {label for label in inferences})
        labels = sorted(labels)

    if len(labels) < 2:
        log.warn(f"not enough unique labels — expecting at least 2 unique labels, received {len(labels)}")
        return None

    matrix = []
    for actual_label in labels:
        matrix.append([confusion_matrix[actual_label][predicted_label] for predicted_label in labels])
    return ConfusionMatrix(title=title, labels=labels, matrix=matrix)


def _roc_curve(y_true_input: List[int], y_score_input: List[float]) -> Tuple[List[float], List[float]]:
    # Convert inputs to numpy arrays
    y_true = np.array(y_true_input)
    y_score = np.array(y_score_input)
    # Sort the predictions by descending order of confidence
    sorted_indices = np.argsort(y_score)[::-1]
    y_score = y_score[sorted_indices]
    y_true = y_true[sorted_indices]
    distinct_value_indices = np.where(np.diff(y_score[sorted_indices]))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
    # Compute the cumulative sums of true positives and false positives
    tps = np.cumsum(y_true)[threshold_idxs]
    fps = np.cumsum(1 - y_true)[threshold_idxs]
    # Drop collinear points (copied from sklearn)
    if len(fps) > 2:
        optimal_indices = np.where(np.r_[True, np.logical_or(np.diff(fps, 2), np.diff(tps, 2)), True])[0]
        fps = fps[optimal_indices]
        tps = tps[optimal_indices]
    fps = np.r_[0, fps]
    tps = np.r_[0, tps]
    # Map cumulative sums to tpr and fpr
    if fps[-1] <= 0:
        # No negative samples in y_true, false positive value should be meaningless
        fpr = []
    else:
        fpr = fps / fps[-1]
        fpr = fpr.tolist()  # type: ignore
    if tps[-1] <= 0:
        # No positive samples in y_true, true positive value should be meaningless
        tpr = []
    else:
        tpr = tps / tps[-1]
        tpr = tpr.tolist()  # type: ignore
    return fpr, tpr


def compute_roc_curves(
    ground_truths: List[Optional[Label]],
    inferences: List[List[ScoredLabel]],
    labels: Optional[List[str]] = None,
    title: Optional[str] = None,
) -> Optional[CurvePlot]:
    """
    Computes OvR (one-vs-rest) ROC (receiver operating characteristic) curves for each class appears in `ground_truths`
    if not specified.

    :param ground_truths: The list of ground truth [`Label`][kolena.workflow.annotation.Label]. For binary
        classification, the negative class can be `None`.
    :param inferences: The list of inference [`ScoredLabel`][kolena.workflow.annotation.ScoredLabel]. For `N`-class
        problems, each inference is expected to contain `N` entries, one for each class and its associated confidence
        score.
    :param labels: The labels to plot. If not specified, classes appear in `ground_truths` are used. Use `labels` to
        specify the evaluating classes especially if `ground_truths` only have negative classes.
    :param title: The title of the plot.
    :return: A [`CurvePlot`][kolena.workflow.plot.CurvePlot] if there is any valid `Curve` computed; otherwise, `None`.
    """
    if len(ground_truths) != len(inferences):
        log.warn(
            f"ground_truths ({len(ground_truths)}) and inferences ({len(inferences)}) differ in length",
        )
        return None

    if len(ground_truths) <= 2:
        log.warn(
            f"insufficient # of samples to compute a roc curve — need at least 3 but received {len(ground_truths)}",
        )
        return None

    if labels is None:
        labels = sorted({gt.label for gt in ground_truths if gt is not None})

    curves: List[Curve] = []
    for label in labels:
        y_true = [1 if gt is not None and gt.label == label else 0 for gt in ground_truths]
        y_score = [get_label_confidence(label, inf) for inf in inferences]
        fpr_values, tpr_values = _roc_curve(y_true_input=y_true, y_score_input=y_score)

        if len(fpr_values) > 0 and len(tpr_values) > 0 and len(fpr_values) == len(tpr_values):
            curves.append(Curve(x=fpr_values, y=tpr_values, label=label))

    inference_labels = [inf.label for inf in inferences[0]]
    is_binary = len(inference_labels) == 1

    if len(curves) > 0:
        if title is None:
            title_prefix = "Receiver Operating Characteristic"
            title = title_prefix if is_binary else title_prefix + " (One-vs-Rest)"
        return CurvePlot(
            title=title,
            x_label="False Positive Rate (FPR)",
            y_label="True Positive Rate (TPR)",
            curves=curves,
        )
    return None


def compute_threshold_curves(
    ground_truths: List[Optional[Label]],
    inferences: List[ScoredLabel],
    thresholds: Optional[List[float]] = None,
) -> Optional[List[Curve]]:
    """
    Computes scores (i.e. [Precision][kolena.workflow.metrics.precision], [Recall][kolena.workflow.metrics.recall]
    and [F1-score][kolena.workflow.metrics.f1_score]) vs. threshold curves for a **single** class presented in
    `inferences`.

    Expects `ground_truths` and `inferences` correspond to the same sample for the same given index.

    :param ground_truths: The list of ground truth [`Label`][kolena.workflow.annotation.Label]s. For binary
        classification, the negative class can be `None`.
    :param inferences: The list of inference [`ScoredLabel`][kolena.workflow.annotation.ScoredLabel]s. The length of
        `inferences` must match the length of `ground_truths`. The list should only include inferences of a specific
        class to plot the threshold curves for.
    :param thresholds: The list of thresholds to plot with. If not specified, all the unique confidence scores are used
        as thresholds, including evenly spaced thresholds from 0 to 1 with 0.1 step.
    :return: A list of [`Curve`][kolena.workflow.plot.Curve]s if there is any valid `Curve` computed; otherwise, `None`.

    """
    if len(ground_truths) != len(inferences):
        log.warn(f"ground_truths ({len(ground_truths)}) and inferences ({len(inferences)}) differ in length")
        return None

    if len(ground_truths) == 0 or len(inferences) == 0:
        log.warn(
            "insufficient # of samples to compute threshold curves — need at least 1 but received "
            f"{len(ground_truths)}",
        )
        return None

    inference_label_list = list({inf.label for inf in inferences})
    if len(inference_label_list) > 1:
        log.warn(
            f"more than one class passed in as inferences: {inference_label_list} — expecting inferences belonging to "
            "a single class",
        )
        return None

    inference_label = inference_label_list[0]
    gts = [gt.label == inference_label if gt else False for gt in ground_truths]

    if thresholds is None:
        unique_scores = list({inf.score for inf in inferences})
        initial_thresholds = list(np.linspace(0, 1, 11))
        thresholds = [0.0]
        for threshold in sorted(set(initial_thresholds + unique_scores)):
            if abs(threshold - thresholds[-1]) >= 1e-2:
                thresholds.append(threshold)
    else:
        thresholds = sorted(set(thresholds))

    precisions = []
    recalls = []
    f1s = []
    for threshold in thresholds:
        infs = [inf.score >= threshold for inf in inferences]
        tp = len([True for gt, inf in zip(gts, infs) if gt and inf])
        fp = len([True for gt, inf in zip(gts, infs) if not gt and inf])
        fn = len([True for gt, inf in zip(gts, infs) if gt and not inf])

        precisions.append(precision(tp, fp))
        recalls.append(recall(tp, fn))
        f1s.append(f1_score(tp, fp, fn))

    return [
        Curve(x=thresholds, y=precisions, label="Precision"),
        Curve(x=thresholds, y=recalls, label="Recall"),
        Curve(x=thresholds, y=f1s, label="F1"),
    ]
