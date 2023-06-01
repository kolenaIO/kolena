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
from typing import Tuple
from typing import Union

import numpy as np
from sklearn.metrics import precision_recall_fscore_support

from kolena.workflow.evaluator import Curve
from kolena.workflow.evaluator import CurvePlot
from kolena.workflow.metrics._geometry import InferenceMatches
from kolena.workflow.metrics._geometry import MulticlassInferenceMatches


def _compute_sklearn_arrays(
    all_matches: List[Union[MulticlassInferenceMatches, InferenceMatches]],
) -> Tuple[List[int], List[int], Dict[str, Tuple[List[int], List[int]]]]:
    arrays_by_label: defaultdict[str, Tuple[List[int], List[int]]] = defaultdict(lambda: ([], []))
    y_true: List[int] = []
    y_score: List[int] = []
    for image_bbox_matches in all_matches:
        for _, bbox_inf in image_bbox_matches.matched:  # TN (if above threshold)
            arrays_by_label[bbox_inf.label][0].append(1)
            arrays_by_label[bbox_inf.label][1].append(bbox_inf.score)
            y_true.append(1)
            y_score.append(bbox_inf.score)
        for bbox_gt, _ in image_bbox_matches.unmatched_gt:  # FN
            y_true.append(1)
            y_score.append(-1)
            arrays_by_label[bbox_gt.label][0].append(1)
            arrays_by_label[bbox_gt.label][1].append(-1)
        for bbox_inf in image_bbox_matches.unmatched_inf:  # FP (if above threshold)
            arrays_by_label[bbox_inf.label][0].append(0)
            arrays_by_label[bbox_inf.label][1].append(bbox_inf.score)
            y_true.append(0)
            y_score.append(bbox_inf.score)
    return y_true, y_score, dict(arrays_by_label)


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


def compute_object_detection_test_case_plots(
    all_matches: List[MulticlassInferenceMatches],
    test_case_name: str,
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

    y_true, y_score, arrays_by_label = _compute_sklearn_arrays(all_matches)
    f1_curve, pr_curve = _compute_threshold_curves(np.array(y_true), np.array(y_score), test_case_name)
    f1_curves.append(f1_curve)
    pr_curves.append(pr_curve)

    classes = sorted(arrays_by_label.keys())
    for label in classes:
        y_true, y_score = arrays_by_label[label]
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
