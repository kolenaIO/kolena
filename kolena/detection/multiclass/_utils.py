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
from typing import Tuple

import numpy as np

from .workflow import GroundTruth
from .workflow import Inference
from .workflow import TestSample
from .workflow import ThresholdConfiguration
from kolena.detection.multiclass.evaluator import threshold_cache
from kolena.detection.multiclass.workflow import ThresholdStrategy
from kolena.workflow.metrics import match_inferences_multiclass

# from kolena._utils import log


def threshold_key(label: str) -> str:
    sanitized = re.sub(r"\W+", "_", label)
    return f"threshold_{sanitized}"


def compute_f1_optimal_thresholds(
    configuration: ThresholdConfiguration,
    inferences: List[Tuple[TestSample, GroundTruth, Inference]],
) -> None:
    if configuration.threshold_strategy != ThresholdStrategy.F1_OPTIMAL:
        return

    if configuration.display_name() in threshold_cache.keys():
        return

    all_bbox_matches = [match_inferences_multiclass(gt, inf, configuration) for _, gt, inf in inferences]
    print(all_bbox_matches)
    labels = {gt.label for _, gts, _ in inferences for gt in gts.bboxes}
    optimal_thresholds: Dict[str, float] = defaultdict(lambda: -1)
    for label in labels:
        y_true, y_score = None, None  # compute_sklearn_arrays(all_bbox_matches, label)
        print(y_true, y_score)
        precision, recall, thresholds = None * 3  # precision_recall_curve(y_true, y_score)
        if thresholds[0] < 0:
            precision = precision[1:]
            recall = recall[1:]
            thresholds = thresholds[1:]
        if len(thresholds) == 0:
            continue
        f1_scores = 2 * precision * recall / (precision + recall)
        max_f1_index = np.argmax(f1_scores)
        optimal_thresholds[label] = thresholds[max_f1_index]

    threshold_cache[configuration.display_name()] = optimal_thresholds
