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
from typing import Union

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

import numpy as np

from kolena._extras.metrics.sklearn import sklearn_metrics
from kolena._utils import log
from kolena.workflow.evaluator import ConfusionMatrix
from kolena.workflow.evaluator import Curve
from kolena.workflow.evaluator import CurvePlot
from kolena.workflow.metrics import InferenceMatches
from kolena.workflow.metrics import MulticlassInferenceMatches


def _compute_sklearn_arrays(
    all_matches: List[Union[MulticlassInferenceMatches, InferenceMatches]],
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


def _compute_threshold_curves(
    y_true: np.ndarray,
    y_score: np.ndarray,
    label: str,
) -> List[Curve]:
    if len(y_score) >= 501:
        thresholds = list(np.linspace(min(abs(y_score)), max(y_score), 501))
    else:
        thresholds = np.unique(y_score[y_score >= 0.0]).tolist()  # sorts

    precisions: List[float] = []
    recalls: List[float] = []
    f1s: List[float] = []
    for threshold in thresholds:
        y_pred = [1 if score >= threshold else 0 for score in y_score]
        precision, recall, f1, _ = sklearn_metrics.precision_recall_fscore_support(
            y_true,
            y_pred,
            average="binary",
            zero_division=0,
        )

        precisions.append(precision)
        recalls.append(recall)
        f1s.append(f1)

    return [Curve(x=recalls, y=precisions, label=label), Curve(x=thresholds, y=f1s, label=label)]


def compute_pr_f1_plots(
    all_matches: List[Union[MulticlassInferenceMatches, InferenceMatches]],
    curve_label: str = "baseline",
    plot: Literal["pr", "f1", "all"] = "all",
) -> List[CurvePlot]:
    """
    Creates a PR (precision and recall) curve and/or F1-threshold (confidence threshold) curve.

    :param all_matches: A list of multiclass or singleclass matching results.
    :param curve_label: The label of the curve.
    :param plot: The specified plot type to return (`pr`, `f1`, or `all`). All plots returned by default.
    :return: :class:`CurvePlot`s for the PR curve and/or F1-threshold curve respectively.
    """
    y_true, y_score = _compute_sklearn_arrays(all_matches)
    curves = _compute_threshold_curves(y_true, y_score, curve_label)
    pr_curves = [curves[0]]
    f1_curves = [curves[1]]

    if plot == "pr":
        pr = CurvePlot(
            title="Precision vs. Recall",
            x_label="Recall",
            y_label="Precision",
            curves=pr_curves,
        )
        return [pr]
    elif plot == "f1":
        threshold_f1 = CurvePlot(
            title="F1-Score vs. Confidence Threshold",
            x_label="Confidence Threshold",
            y_label="F1-Score",
            curves=f1_curves,
        )
        return [threshold_f1]

    pr = CurvePlot(
        title="Precision vs. Recall",
        x_label="Recall",
        y_label="Precision",
        curves=pr_curves,
    )

    threshold_f1 = CurvePlot(
        title="F1-Score vs. Confidence Threshold",
        x_label="Confidence Threshold",
        y_label="F1-Score",
        curves=f1_curves,
    )

    return [pr, threshold_f1]


def compute_confusion_matrix_plot(
    all_matches: List[MulticlassInferenceMatches],
    plot_title: str = "Confusion Matrix",
) -> Optional[ConfusionMatrix]:
    """
    Creates a confusion matrix for the multiclass object detection workflow.

    :param all_matches: A list of multiclass matching results.
    :param plot_title: The title for the :class:`ConfusionMatrix`.
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

    if len(labels) < 2:
        log.info(f"skipping confusion matrix for a single label: {labels}")
        return None

    ordered_labels = sorted(labels)
    matrix = []
    for actual_label in ordered_labels:
        matrix.append([confusion_matrix[actual_label][predicted_label] for predicted_label in ordered_labels])
    return ConfusionMatrix(title=plot_title, labels=ordered_labels, matrix=matrix)
