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
import re
from collections import defaultdict
from typing import Dict
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import Union

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from kolena._utils import log
from kolena.workflow.evaluator import ConfusionMatrix
from kolena.workflow.evaluator import Curve
from kolena.workflow.evaluator import CurvePlot
from kolena.workflow.metrics._geometry import InferenceMatches
from kolena.workflow.metrics._geometry import MulticlassInferenceMatches


def threshold_key(label: str) -> str:
    sanitized = re.sub(r"\W+", "_", label)
    return f"threshold_{sanitized}"


def _compute_sklearn_arrays(
    all_matches: List[Union[MulticlassInferenceMatches, InferenceMatches]],
) -> Tuple[List[int], List[int], Dict[str, List[int]], Dict[str, List[int]]]:
    y_true_by_label: defaultdict[str, List[int]] = defaultdict(lambda: [])
    y_score_by_label: defaultdict[str, List[int]] = defaultdict(lambda: [])
    y_true: List[int] = []
    y_score: List[int] = []
    for image_bbox_matches in all_matches:
        for _, bbox_inf in image_bbox_matches.matched:  # TN (if above threshold)
            y_true_by_label[bbox_inf.label].append(1)
            y_score_by_label[bbox_inf.label].append(bbox_inf.score)
            y_true.append(1)
            y_score.append(bbox_inf.score)
        for bbox_gt, _ in image_bbox_matches.unmatched_gt:  # FN
            y_true.append(1)
            y_score.append(-1)
            y_true_by_label[bbox_gt.label].append(1)
            y_score_by_label[bbox_gt.label].append(-1)
        for bbox_inf in image_bbox_matches.unmatched_inf:  # FP (if above threshold)
            y_true_by_label[bbox_inf.label].append(0)
            y_score_by_label[bbox_inf.label].append(bbox_inf.score)
            y_true.append(0)
            y_score.append(bbox_inf.score)
    return y_true, y_score, dict(y_true_by_label), dict(y_score_by_label)


def _compute_threshold_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str,
) -> Tuple[Curve, Curve]:
    thresholds = list(np.linspace(min(abs(y_score)), max(y_score), min(401, len(y_score))))[:-1]
    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for threshold in thresholds:
        y_pred = [1 if score > threshold else 0 for score in y_score]
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )
        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)
    return Curve(x=thresholds, y=f1s, label=label), Curve(x=recalls, y=precisions, label=label)


def compute_pr_f1_plots(
    all_matches: List[MulticlassInferenceMatches],
    plot_title: str = "pr f1 plots",
) -> List[CurvePlot]:
    """
    Creates a PR (precision and recall) curve and F1-threshold (confidence threshold) curve for the multiclass object
    detection workflow. For `n` labels, each plot has `n+1` curves. One for the test case, and one per label.

    :param all_matches: a list of multiclass or single class matching results.
    :param test_case_name: the name of the test case.
    :return: Two :class:`CurvePlot`s for the PR curves and F1-threshold curves for the test case and each label.
    """
    f1_curves: List[Curve] = []
    pr_curves: List[Curve] = []

    y_true, y_score, y_true_by_label, y_score_by_label = _compute_sklearn_arrays(all_matches)
    f1_curve, pr_curve = _compute_threshold_curves(np.array(y_true), np.array(y_score), plot_title)
    f1_curves.append(f1_curve)
    pr_curves.append(pr_curve)

    classes = sorted(y_true_by_label.keys())
    for label in classes:
        y_true, y_score = y_true_by_label[label], y_score_by_label[label]
        if len(y_true) > 0:
            f1_curve, pr_curve = _compute_threshold_curves(np.array(y_true), np.array(y_score), label)
            f1_curves.append(f1_curve)
            pr_curves.append(pr_curve)

    threshold_f1 = CurvePlot(
        title="F1-Score vs. Confidence Threshold",
        x_label="Confidence Threshold",
        y_label="F1-Score",
        curves=f1_curves,
    )

    pr = CurvePlot(
        title="Precision vs. Recall",
        x_label="Recall",
        y_label="Precision",
        curves=pr_curves,
    )

    return [threshold_f1, pr]


def compute_confusion_matrix_plot(
    all_matches: List[MulticlassInferenceMatches],
    plot_title: str = "Confusion Matrix",
) -> Optional[ConfusionMatrix]:
    """
    Creates a confusion matrix for the multiclass object detection workflow.

    :param all_matches: a list of multiclass matching results.
    :param plot_title: the title for the :class:`ConfusionMatrix`.
    :return: :class:`ConfusionMatrix` with all actual and predicted labels, if there is more than one label.
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

        labels.update(inf.label for inf in match.unmatched_inf)

    if len(labels) < 2:
        log.info(f"skipping confusion matrix for a single label: {labels}")
        return None

    ordered_labels = sorted(labels)
    matrix = []
    for actual_label in ordered_labels:
        matrix.append([confusion_matrix[actual_label][predicted_label] for predicted_label in ordered_labels])
    return ConfusionMatrix(title=plot_title, labels=ordered_labels, matrix=matrix)


def compute_ap(precisions: List[float], recalls: List[float]) -> float:
    """
    Computes Average Precision given a PR curve using the same approach as Pascal VOC.
    Based on https://github.com/Cartucho/mAP which implements the Matlab Pascal Voc code in Python.
    """

    if len(precisions) != len(recalls):
        raise ValueError("precisions and recalls differ in length")

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
