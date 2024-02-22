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
from typing import Literal
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np

from kolena._extras.metrics.sklearn import sklearn_metrics
from kolena._utils import log
from kolena.workflow import ConfusionMatrix
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledPolygon
from kolena.workflow.metrics import InferenceMatches
from kolena.workflow.metrics import MulticlassInferenceMatches


def filter_inferences(
    inferences: List[Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon]],
    confidence_score: Optional[float] = None,
    labels: Optional[Set[str]] = None,
) -> List[Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon]]:
    filtered_by_confidence = (
        [inf for inf in inferences if inf.score >= confidence_score] if confidence_score else inferences
    )
    if labels is None:
        return filtered_by_confidence
    return [inf for inf in filtered_by_confidence if inf.label in labels]


def _compute_sklearn_arrays(
    all_matches: Union[List[MulticlassInferenceMatches], List[InferenceMatches]],
) -> Tuple[np.ndarray, np.ndarray]:
    y_true: List[int] = []
    y_score: List[float] = []
    for image_bbox_matches in all_matches:
        for _, bbox_inf in image_bbox_matches.matched:  # TP (if above threshold)
            y_true.append(1)
            y_score.append(bbox_inf.score)
        for _ in image_bbox_matches.unmatched_gt:  # FN
            y_true.append(1)
            y_score.append(-1)
        for bbox_inf in image_bbox_matches.unmatched_inf:  # FP (if above threshold)
            y_true.append(0)
            y_score.append(bbox_inf.score)
    return np.array(y_true), np.array(y_score)


def _compute_threshold_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
    curve_type: Literal["pr", "f1"],
    curve_label: Optional[str] = None,
) -> Optional[Curve]:
    if len(y_score) < 1:
        return None

    potential_thresholds = np.unique(y_score[y_score >= 0.0]).tolist()  # sorts

    if len(potential_thresholds) >= 501:
        potential_thresholds = list(np.linspace(min(potential_thresholds), max(potential_thresholds), 501))

    precisions: List[float] = []
    recalls: List[float] = []
    thresholds: List[float] = []
    f1s: List[float] = []
    for threshold in potential_thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_score]
        precision, recall, f1, _ = sklearn_metrics.precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        # avoid curves with one x-value and two y-values
        if recall in recalls:
            idx = recalls.index(recall)
            precisions[idx] = max(precisions[idx], precision)
            # if an old point is replaced, replace the f1s and thresholds
            if precisions[idx] == precision:
                thresholds[idx] = threshold
                f1s[idx] = f1
        else:
            precisions.append(precision)
            recalls.append(recall)
            thresholds.append(threshold)
            f1s.append(f1)

    # omit curves with no points
    if len(f1s) == 0 or len(precisions) == 0 or len(recalls) == 0:
        return None

    if curve_type == "f1":
        return (
            Curve(x=thresholds, y=f1s, label=curve_label, extra=dict(Precision=precisions, Recall=recalls))
            if len(f1s) >= 2
            else None
        )

    # add a point to start the PR curve on the vertical axis if needed
    if 0.0 not in recalls:
        minpos = recalls.index(min(recalls))
        precisions.append(precisions[minpos])
        recalls.append(0.0)
        # maintain the same lengths for f1s and thresholds
        f1s.append(f1s[minpos])
        thresholds.append(thresholds[minpos])
    return (
        Curve(x=recalls, y=precisions, label=curve_label, extra=dict(F1=f1s, Threshold=thresholds))
        if len(recalls) >= 2
        else None
    )


def _compute_multiclass_curves(
    all_matches: List[MulticlassInferenceMatches],
    curve_type: Literal["pr", "f1"],
) -> Optional[List[Curve]]:
    curves: List[Curve] = []
    y_true_score_by_label = _compute_sklearn_arrays_by_class(all_matches)

    for label, (y_true, y_score) in sorted(y_true_score_by_label.items(), key=lambda x: x[0]):
        curve = _compute_threshold_curve(y_true, y_score, curve_type, label)
        if curve is not None:
            curves.append(curve)
    return curves if curves else None


def compute_pr_curve(
    all_matches: Union[List[MulticlassInferenceMatches], List[InferenceMatches]],
    curve_label: Optional[str] = None,
) -> Optional[Curve]:
    """
    Creates a PR (precision and recall) curve.

    :param all_matches: A list of multiclass or singleclass matching results.
    :param curve_label: The label of the curve.
    :return: :class:`Curve` for the PR curve.
    """
    y_true, y_score = _compute_sklearn_arrays(all_matches)
    return _compute_threshold_curve(y_true, y_score, "pr", curve_label)


def compute_pr_plot(
    all_matches: Union[List[MulticlassInferenceMatches], List[InferenceMatches]],
    curve_label: Optional[str] = None,
) -> Optional[CurvePlot]:
    """
    Creates a PR (precision and recall) plot.

    :param all_matches: A list of multiclass or singleclass matching results.
    :param curve_label: The label of the curve.
    :return: :class:`CurvePlot` for the PR curve.
    """
    curve = compute_pr_curve(all_matches, curve_label)

    return (
        CurvePlot(
            title="Precision vs. Recall",
            x_label="Recall",
            y_label="Precision",
            curves=[curve],
        )
        if curve
        else None
    )


def compute_pr_plot_multiclass(
    all_matches: List[MulticlassInferenceMatches],
) -> Optional[CurvePlot]:
    """
    Creates a PR (precision-recall) curve for the multiclass object detection workflow.
    For `n` labels, each plot has `n+1` curves. One for the test case, and one per label.

    :param all_matches: a list of multiclass matching results.
    :return: :class:`CurvePlot` for the PR curves of the test case and each label.
    """
    pr_curves: Optional[List[Curve]] = _compute_multiclass_curves(all_matches, "pr")

    return (
        CurvePlot(
            title="Precision vs. Recall Per Class",
            x_label="Recall",
            y_label="Precision",
            curves=pr_curves,
        )
        if pr_curves
        else None
    )


def compute_f1_plot(
    all_matches: Union[List[MulticlassInferenceMatches], List[InferenceMatches]],
    curve_label: Optional[str] = None,
) -> Optional[CurvePlot]:
    """
    Creates a F1-threshold (confidence threshold) plot.

    :param all_matches: A list of multiclass or singleclass matching results.
    :param curve_label: The label of the curve.
    :return: :class:`CurvePlot` for the F1-threshold curve.
    """
    y_true, y_score = _compute_sklearn_arrays(all_matches)
    curve = _compute_threshold_curve(y_true, y_score, "f1", curve_label)

    return (
        CurvePlot(
            title="F1-Score vs. Confidence Threshold",
            x_label="Confidence Threshold",
            y_label="F1-Score",
            curves=[curve],
        )
        if curve
        else None
    )


def compute_f1_plot_multiclass(
    all_matches: List[MulticlassInferenceMatches],
) -> Optional[CurvePlot]:
    """
    Creates a F1-threshold (confidence threshold) curve for the multiclass object detection workflow.
    For `n` labels, each plot has `n+1` curves. One for the test case, and one per label.

    :param all_matches: a list of multiclass matching results.
    :param curve_label: the label of the main curve.
    :return: :class:`CurvePlot` for the F1-threshold curves of the test case and each label.
    """
    f1_curves: Optional[List[Curve]] = _compute_multiclass_curves(all_matches, "f1")

    return (
        CurvePlot(
            title="F1-Score vs. Confidence Threshold Per Class",
            x_label="Confidence Threshold",
            y_label="F1-Score",
            curves=f1_curves,
        )
        if f1_curves
        else None
    )


def compute_confusion_matrix_plot(
    all_matches: List[MulticlassInferenceMatches],
    plot_title: str = "Confusion Matrix",
) -> Optional[ConfusionMatrix]:
    """
    Creates a [`ConfusionMatrix`][kolena.workflow.ConfusionMatrix] for a multiclass workflow.

    :param all_matches: A list of multiclass matching results.
    :param plot_title: The title for the [`ConfusionMatrix`][kolena.workflow.ConfusionMatrix].
    :return: [`ConfusionMatrix`][kolena.workflow.ConfusionMatrix] with all actual and predicted labels, if there is more
        than one label in the provided `all_matches`.
    """
    labels: Set[str] = set()

    confusion_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for match in all_matches:
        for gt, _ in match.matched:
            actual_label = gt.label
            confusion_matrix[actual_label][actual_label] += 1
            labels.add(actual_label)

        for gt, inf in match.unmatched_gt:
            actual_label = gt.label
            labels.add(actual_label)
            if inf is not None:
                predicted_label = inf.label
                confusion_matrix[actual_label][predicted_label] += 1
                labels.add(predicted_label)

    if len(labels) < 2:
        log.info(f"skipping confusion matrix for a single label: {labels}")
        return None

    ordered_labels = sorted(labels)
    matrix = []
    for actual_label in ordered_labels:
        matrix.append([confusion_matrix[actual_label][predicted_label] for predicted_label in ordered_labels])
    return ConfusionMatrix(title=plot_title, labels=ordered_labels, matrix=matrix)


def _compute_sklearn_arrays_by_class(
    all_matches: List[MulticlassInferenceMatches],
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    y_true_and_score_by_label: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    labels: Set[str] = set()
    for match in all_matches:
        for _, bbox_inf in match.matched:
            labels.add(bbox_inf.label)
        for bbox_gt, _ in match.unmatched_gt:
            labels.add(bbox_gt.label)
        for bbox_inf in match.unmatched_inf:
            labels.add(bbox_inf.label)

    for label in labels:
        filtered_matchings: List[InferenceMatches] = [
            InferenceMatches(
                matched=[(gt, inf) for gt, inf in match.matched if gt.label == label],
                unmatched_gt=[gt for gt, _ in match.unmatched_gt if gt.label == label],
                unmatched_inf=[inf for inf in match.unmatched_inf if inf.label == label],
            )
            for match in all_matches
        ]

        y_true, y_score = _compute_sklearn_arrays(filtered_matchings)
        y_true_and_score_by_label[label] = (y_true, y_score)

    return y_true_and_score_by_label


def _compute_optimal_f1_with_arrays(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> float:
    # optimal threshold is 0 if there is no relevant or true label
    if len(y_true) == 0 or np.all(y_true == 0):
        return 0.0

    precision, recall, thresholds = sklearn_metrics.precision_recall_curve(y_true, y_score)

    # delete last pr of (1,0)
    precision = precision[:-1]
    recall = recall[:-1]

    if thresholds[0] < 0:
        precision = precision[1:]
        recall = recall[1:]
        thresholds = thresholds[1:]
    if len(thresholds) == 0:
        return 0.0

    f1_scores = 2 * precision * recall / (precision + recall)
    f1_scores = np.nan_to_num(f1_scores, nan=0)
    max_f1_index = np.argmax(f1_scores)
    if f1_scores[max_f1_index] == 0:
        return 0.0
    else:
        return float(thresholds[max_f1_index])


def compute_optimal_f1_threshold(
    all_matches: Union[List[MulticlassInferenceMatches], List[InferenceMatches]],
) -> float:
    """
    Computes the optimal F1 threshold for matchings.

    :param all_matches: A list of matching results.
    :return: The optimal F1 threshold value, zero where invalid.
    """
    y_true, y_score = _compute_sklearn_arrays(all_matches)
    return _compute_optimal_f1_with_arrays(y_true, y_score)


def compute_optimal_f1_threshold_multiclass(
    all_matches: List[MulticlassInferenceMatches],
) -> Dict[str, float]:
    """
    Creates a mapping of label to optimal F1 thresholds for a multiclass workflow.

    :param all_matches: A list of multiclass matching results.
    :return: A dictionary with each label and its optimal F1 threshold value, zero where invalid.
    """
    optimal_thresholds: Dict[str, float] = {}

    y_true_score_by_label = _compute_sklearn_arrays_by_class(all_matches)
    for label, (y_true, y_score) in sorted(y_true_score_by_label.items(), key=lambda x: x[0]):
        optimal_thresholds[label] = _compute_optimal_f1_with_arrays(y_true, y_score)
    return optimal_thresholds


def compute_average_precision(precisions: List[float], recalls: List[float]) -> float:
    """
    Computes the average precision given a PR curve with the metrics methodology of PASCAL VOC.
    Based on the [PASCAL VOC code in Python](https://github.com/Cartucho/mAP).

    :param precisions: A list precision values from a PR curve.
    :param recalls: A list recall values from a PR curve.
    :return: The value of the average precision.
    """

    if len(precisions) != len(recalls):
        raise ValueError("precisions and recalls differ in length")

    if len(precisions) == 0:
        return 0

    pairs = sorted(zip(recalls, precisions), key=lambda x: x[0])
    recalls = [x[0] for x in pairs]
    precisions = [x[1] for x in pairs]
    # add (0,0) to left and (1,0) to right
    recalls = [0, *recalls, 1]
    precisions = [0, *precisions, 0]

    # make precisions monotonic decreasing
    for i in range(len(precisions) - 2, -1, -1):
        precisions[i] = max(precisions[i], precisions[i + 1])

    # indices where recall has changed
    recall_changed_indices = []
    for i in range(1, len(recalls)):
        if recalls[i] != recalls[i - 1]:
            recall_changed_indices.append(i)

    ap = 0.0
    for i in recall_changed_indices:
        ap += (recalls[i] - recalls[i - 1]) * precisions[i]
    return ap
