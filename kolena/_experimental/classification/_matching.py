# Copyright 2021-2025 Kolena Inc.
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
from typing import Generic
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena.annotation import Label
from kolena.annotation import ScoredLabel

GT = TypeVar("GT", bound=Union[str, Label])
Inf = TypeVar("Inf", bound=Union[str, Label, ScoredLabel])


@dataclass(frozen=True)
class InferenceMatches(Generic[GT, Inf]):
    """
    The result of [`match_inferences`][kolena._experimental.classification._matching.match_inferences], providing lists
    of matches between ground truth and inference objects, unmatched ground truths, and unmatched inferences.
    `InferenceMatches` can be used to calculate metrics such as precision and recall.

    Objects are of type `str` or [`Label`][kolena.annotation.Label], depending on the type of inputs provided to
    [`match_inferences`][kolena.metrics.match_inferences].
    """

    matched: List[Tuple[GT, Inf]]
    """
    Pairs of matched ground truth and inference objects. Considered as true positives.
    """

    unmatched_gt: List[GT]
    """Unmatched ground truth objects. Considered as false negatives."""

    unmatched_inf: List[Inf]
    """
    Unmatched inference objects. Considered as false positives.
    """


def _get_keyed_items(
    items: List,
    required_match_fields: List[str],
) -> Dict[Tuple, List]:
    keyed_items = defaultdict(list)
    for item in items:
        key = tuple(getattr(item, field, None) for field in required_match_fields)
        keyed_items[key].append(item)
    return keyed_items


def match_inferences(
    ground_truths: List[GT],
    inferences: List[Inf],
    *,
    ignored_ground_truths: Optional[List[GT]] = None,
    required_match_fields: Optional[List[str]] = None,
) -> InferenceMatches[GT, Inf]:
    if required_match_fields is None or len(required_match_fields) == 0:
        return _match_inferences(
            ground_truths,
            inferences,
            ignored_ground_truths=ignored_ground_truths,
        )

    keyed_inferences = _get_keyed_items(inferences, required_match_fields)
    keyed_ground_truths = _get_keyed_items(ground_truths, required_match_fields)
    keyed_ignore_ground_truths = (
        _get_keyed_items(ignored_ground_truths, required_match_fields) if ignored_ground_truths else defaultdict(list)
    )
    keys = {*keyed_inferences.keys(), *keyed_ground_truths.keys(), *keyed_ignore_ground_truths.keys()}
    inf_matches = [
        _match_inferences(
            keyed_ground_truths[key],
            keyed_inferences[key],
            ignored_ground_truths=keyed_ignore_ground_truths[key],
        )
        for key in keys
    ]
    flattened_matched: List[Tuple[GT, Inf]] = []
    flattened_unmatched_gt: List[GT] = []
    flattened_unmatched_inf: List[Inf] = []
    for inf_match in inf_matches:
        flattened_matched.extend(inf_match.matched)
        flattened_unmatched_gt.extend(inf_match.unmatched_gt)
        flattened_unmatched_inf.extend(inf_match.unmatched_inf)
    return InferenceMatches(
        matched=flattened_matched,
        unmatched_gt=flattened_unmatched_gt,
        unmatched_inf=flattened_unmatched_inf,
    )


def _match_inferences(
    ground_truths: List[GT],
    inferences: List[Inf],
    *,
    ignored_ground_truths: Optional[List[GT]] = None,
) -> InferenceMatches[GT, Inf]:
    inferences = sorted(inferences, key=lambda inf: inf.score if hasattr(inf, "score") else 0, reverse=True)
    matched: List[Tuple[GT, Inf]] = []
    unmatched_inf: List[Inf] = []
    taken_gts: Set[int] = set()

    gt_objects = ground_truths
    if ignored_ground_truths:
        gt_objects = gt_objects + ignored_ground_truths

    for inf in inferences:
        is_matched = False
        for g, gt in enumerate(gt_objects):
            if g in taken_gts:
                continue
            if get_label(gt) == get_label(inf):
                if g < len(ground_truths):
                    matched.append((gt, inf))
                taken_gts.add(g)
                is_matched = True
                break
        if not is_matched:
            unmatched_inf.append(inf)
    unmatched_gt = [gt for g, gt in enumerate(ground_truths) if g not in taken_gts]
    return InferenceMatches(matched=matched, unmatched_gt=unmatched_gt, unmatched_inf=unmatched_inf)


def get_label(classification: Union[str, Label, ScoredLabel]) -> str:
    if isinstance(classification, str):
        return classification
    return classification.label
