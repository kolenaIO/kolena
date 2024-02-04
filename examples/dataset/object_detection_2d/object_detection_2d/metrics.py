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
from typing import Any
from typing import Dict

from kolena.annotation import ScoredLabel
from kolena.workflow.metrics import MulticlassInferenceMatches


def test_sample_metrics(
    bbox_matches: MulticlassInferenceMatches,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    tp = [inf for _, inf in bbox_matches.matched if inf.score >= thresholds[inf.label]]
    fp = [inf for inf in bbox_matches.unmatched_inf if inf.score >= thresholds[inf.label]]
    fn = [gt for gt, _ in bbox_matches.unmatched_gt] + [
        gt for gt, inf in bbox_matches.matched if inf.score < thresholds[inf.label]
    ]
    confused = [inf for _, inf in bbox_matches.unmatched_gt if inf is not None and inf.score >= thresholds[inf.label]]
    non_ignored_inferences = tp + fp
    scores = [inf.score for inf in non_ignored_inferences]
    inference_labels = {inf.label for _, inf in bbox_matches.matched} | {
        inf.label for inf in bbox_matches.unmatched_inf
    }
    fields = [
        ScoredLabel(label=label, score=thresholds[label])
        for label in sorted(thresholds.keys())
        if label in inference_labels
    ]
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "Confused": confused,
        "count_TP": len(tp),
        "count_FP": len(fp),
        "count_FN": len(fn),
        "count_Confused": len(confused),
        "has_TP": len(tp) > 0,
        "has_FP": len(fp) > 0,
        "has_FN": len(fn) > 0,
        "has_Confused": len(confused) > 0,
        "ignored": False,
        "max_confidence_above_t": max(scores) if len(scores) > 0 else None,
        "min_confidence_above_t": min(scores) if len(scores) > 0 else None,
        "thresholds": fields,
    }
