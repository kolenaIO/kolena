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
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np

from kolena._utils import log
from kolena.classification.multiclass import InferenceLabel
from kolena.workflow.annotation import ScoredClassificationLabel


def get_label_confidence(label: str, inference_labels: List[Union[ScoredClassificationLabel, InferenceLabel]]) -> float:
    for inf_label in inference_labels:
        if inf_label.label == label:
            return inf_label.score
    return 0


def roc_curve(y_true: List[int], y_score: List[float]) -> Tuple[List[float], List[float]]:
    # Convert inputs to numpy arrays
    y_true = np.array(y_true)
    y_score = np.array(y_score)

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
        fpr = fpr.tolist()
    if tps[-1] <= 0:
        # No positive samples in y_true, true positive value should be meaningless
        tpr = []
    else:
        tpr = tps / tps[-1]
        tpr = tpr.tolist()

    return fpr, tpr


def get_histogram_range(scores: List[float]) -> Optional[Tuple[float, float, int]]:  # min, max, num_buckets
    min_score, max_score = min(scores), max(scores)
    if min_score < 0.0 or max_score > 1.0:
        log.warn(
            f"scores out of range for confidence histograms: expecting [0, 1], got [{min_score:.3f}, {max_score:.3f}]",
        )
        return None

    bin_range_options = np.linspace(0, 1, 51)
    min_range, max_range, bucket_fenceposts = 0, 1, 0
    for option in bin_range_options:
        if min_score >= option > min_range:
            min_range = option
        if max_score <= option < max_range:
            max_range = option
    for option in bin_range_options:
        if min_range <= option <= max_range:
            bucket_fenceposts += 1
    return min_range, max_range, bucket_fenceposts - 1
