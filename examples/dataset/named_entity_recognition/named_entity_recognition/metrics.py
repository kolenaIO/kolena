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
from typing import Dict
from typing import Iterable
from typing import List
from typing import Set
from typing import Tuple

from kolena.annotation import LabeledTextSegment


def find_overlap(true_range: Iterable, pred_range: Iterable) -> Set:
    """Find the overlap between two ranges

    Find the overlap between two ranges. Return the overlapping values if
    present, else return an empty set().

    Examples:

    >>> find_overlap((1, 2), (2, 3))
    2
    >>> find_overlap((1, 2), (3, 4))
    set()
    """

    true_set = set(true_range)
    pred_set = set(pred_range)

    overlaps = true_set.intersection(pred_set)

    return overlaps


def evaluate(
    groundtruths: List[LabeledTextSegment],
    inferences: List[LabeledTextSegment],
    tags: Iterable[str],
) -> Tuple[List[LabeledTextSegment], List[LabeledTextSegment], List[LabeledTextSegment], Dict[str, int]]:
    TP: List[LabeledTextSegment] = []
    FP: List[LabeledTextSegment] = []
    FN: List[LabeledTextSegment] = []

    metrics = {"TP": 0, "FP": 0, "FN": 0, "CLS_ERROR": 0, "LOC_ERROR": 0, "OVR_GEN": 0, "GT": 0, "INF": 0}

    # Subset into only the tags that we are interested in.
    # NOTE: we remove the tags we don't want from both the predicted and the
    # true entities.
    true_named_entities = [ent for ent in groundtruths if ent.label in tags]
    pred_named_entities = [ent for ent in inferences if ent.label in tags]

    # go through each predicted named-entity
    for pred in pred_named_entities:
        found_overlap = False

        # Scenario 1: exact match between true and pred (TP)
        if pred in true_named_entities:
            metrics["TP"] += 1
            TP.append(pred)
        else:
            # check for overlaps with any of the true entities
            for true in true_named_entities:
                pred_range = range(pred.start, pred.end)
                true_range = range(true.start, true.end)

                # Scenario 2: offsets match, but entity type is wrong (FP - classification error)
                if true.start == pred.start and pred.end == true.end and true.label != pred.label:
                    metrics["FP"] += 1
                    metrics["CLS_ERROR"] += 1
                    FP.append(pred)

                    found_overlap = True
                    break

                # check for an overlap i.e. not exact boundary match, with true entities
                elif find_overlap(true_range, pred_range):
                    # Scenario 3: there is an overlap (but offsets do not match
                    # exactly), and the entity type is the same. (FP - localization error)
                    if pred.label == true.label:
                        metrics["FP"] += 1
                        metrics["LOC_ERROR"] += 1
                        FP.append(pred)

                        found_overlap = True
                        break

                    # Scenario 4: entities overlap, but the entity type is
                    # different. (FP)
                    else:
                        metrics["FP"] += 1
                        metrics["CLS_ERROR"] += 1
                        metrics["LOC_ERROR"] += 1
                        FP.append(pred)

                        found_overlap = True
                        break

            # Scenario 5: entities are spurious (i.e., over-generated). (FP - over-generated)
            if not found_overlap:
                metrics["FP"] += 1
                metrics["OVR_GEN"] += 1
                FP.append(pred)

    # Scenario 6: entity was missed entirely.
    for true in true_named_entities:
        if true in TP:
            continue
        else:
            metrics["FN"] += 1
            FN.append(true)

    metrics["GT"] = len(groundtruths)
    metrics["INF"] = len(inferences)

    return TP, FP, FN, metrics
