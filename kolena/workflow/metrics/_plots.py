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
from typing import Union

from kolena.workflow.evaluator import ConfusionMatrix
from kolena.workflow.evaluator import Plot
from kolena.workflow.metrics._geometry import InferenceMatches
from kolena.workflow.metrics._geometry import MulticlassInferenceMatches


def compute_test_case_confusion_matrix(
    all_matches: List[Union[MulticlassInferenceMatches, InferenceMatches]],
    test_case_name: str,
    plot_title: str,
) -> Optional[Plot]:
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
        print(f"skipping confusion matrix for {test_case_name}: single label test case")
        return None

    if len(labels) > 10:
        print(f"skipping confusion matrix for {test_case_name}: too many labels")
        return None

    ordered_labels = sorted(labels)
    matrix = []
    for actual_label in ordered_labels:
        matrix.append([confusion_matrix[actual_label][predicted_label] for predicted_label in ordered_labels])
    return ConfusionMatrix(title=plot_title, labels=ordered_labels, matrix=matrix)
