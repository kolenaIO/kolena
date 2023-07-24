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
from typing import Tuple

import numpy as np

from kolena._utils import log
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.plot import ConfusionMatrix
from kolena.workflow.plot import Histogram


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


def compute_confusion_matrix(
    gts: List[str],
    infs: List[str],
    title: Optional[str] = "Confusion Matrix",
    labels: Optional[List[str]] = None,
) -> Optional[ConfusionMatrix]:
    """
    Computes confusion matrix given a list of ground truth and inference labels.

    For a binary classification case, a 2x2 confusion matrix with the count of TP, FP, FN, and TP is computed.

    :param gts: The ground truth labels
    :param infs: The inference labels
    :param title: The title of confusion matrix
    :param labels: The list of labels to index the matrix. This may be used to reorder or select a subset of labels.
                   By default, labels that appear at least once in `gts` or `infs` are used in sorted order.
    :return: The confusion matrix.

    """
    if len(gts) != len(infs):
        log.warn(
            f"ground truth labels ({len(gts)}) and inference labels ({len(infs)}) "
            "differ in length for a confusion matrix — expecting equal length",
        )
        return None

    if len(gts) == 0:
        log.warn("no ground_truths provided for a confusion matrix")
        return None

    confusion_matrix: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))

    for gt, inf in zip(gts, infs):
        confusion_matrix[gt][inf] += 1

    if labels is None:
        labels = list({label for label in gts} | {label for label in infs})
        labels = sorted(labels)

    if len(labels) < 2:
        log.warn(f"not enough unique labels — expecting at least 2 unique labels, received {len(labels)}")
        return None

    matrix = []
    for actual_label in labels:
        matrix.append([confusion_matrix[actual_label][predicted_label] for predicted_label in labels])
    return ConfusionMatrix(title=title, labels=labels, matrix=matrix)
