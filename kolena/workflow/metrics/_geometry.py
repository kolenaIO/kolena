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
from dataclasses import dataclass
from typing import Generic
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal

from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import Polygon
from kolena.workflow.annotation import ScoredBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledPolygon
from kolena.workflow.annotation import ScoredPolygon


def _iou_bbox(box1: BoundingBox, box2: BoundingBox) -> float:
    # get coordinates of the intersection rectangle
    x1_inter = max(box1.top_left[0], box2.top_left[0])
    y1_inter = max(box1.top_left[1], box2.top_left[1])
    x2_inter = min(box1.bottom_right[0], box2.bottom_right[0])
    y2_inter = min(box1.bottom_right[1], box2.bottom_right[1])

    # check if there is an intersection
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0

    intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)

    if intersection <= 0:
        return 0.0

    box1_width = box1.bottom_right[0] - box1.top_left[0]
    box1_height = box1.bottom_right[1] - box1.top_left[1]
    box1_area = box1_width * box1_height

    box2_width = box2.bottom_right[0] - box2.top_left[0]
    box2_height = box2.bottom_right[1] - box2.top_left[1]
    box2_area = box2_width * box2_height

    union = box1_area + box2_area - intersection
    iou = intersection / union if union > 0 else 0.0
    return iou


def iou(a: Union[BoundingBox, Polygon], b: Union[BoundingBox, Polygon]) -> float:
    """
    Compute the Intersection Over Union (IOU) of two geometries.

    :param a: the first geometry in computation.
    :param b: the second geometry in computation.
    :return: the value of the IOU between geometries `a` and `b`.
    """

    if isinstance(a, BoundingBox) and isinstance(b, BoundingBox):
        return _iou_bbox(a, b)

    def as_shapely_polygon(obj: Union[BoundingBox, Polygon]) -> ShapelyPolygon:
        if isinstance(obj, BoundingBox):
            (tlx, tly), (brx, bry) = obj.top_left, obj.bottom_right
            return make_valid(ShapelyPolygon([(tlx, tly), (brx, tly), (brx, bry), (tlx, bry)]))
        return make_valid(ShapelyPolygon(obj.points))

    polygon_a = as_shapely_polygon(a)
    polygon_b = as_shapely_polygon(b)
    union = polygon_a.union(polygon_b).area
    return polygon_a.intersection(polygon_b).area / union if union > 0 else 0


GT = TypeVar("GT", bound=Union[BoundingBox, Polygon, LabeledBoundingBox, LabeledPolygon])
Inf = TypeVar("Inf", bound=Union[ScoredBoundingBox, ScoredPolygon, ScoredLabeledBoundingBox, ScoredLabeledPolygon])


@dataclass(frozen=True)
class InferenceMatches(Generic[GT, Inf]):
    """
    The result of :func:`match_inferences`, providing lists of matches between ground truth and inference objects,
    unmatched ground truths, and unmatched inferences. :class:`InferenceMatches` can be used to calculate metrics such
    as precision and recall.

    Objects must be a BoundingBox or a Polygon.

    :attribute GT: A union of bounding box and polygon classes with or without a label
    :attribute Inf: A union of scored bounding box and polygon classes with or without a label
    """

    #: Pairs of matched ground truth and inference objects above the IOU threshold. True positives.
    matched: List[Tuple[GT, Inf]]
    #: Unmatched ground truth objects. False negatives.
    unmatched_gt: List[GT]
    #: Unmatched inference objects. False positives.
    unmatched_inf: List[Inf]


def _match_inferences_single_class_pascal_voc(
    ground_truths: List[GT],
    inferences: List[Inf],
    ignored_ground_truths: Optional[List[GT]] = None,
    iou_threshold: float = 0.5,
) -> InferenceMatches[GT, Inf]:
    matched: List[Tuple[GT, Inf]] = []
    unmatched_gt: List[GT] = []
    unmatched_inf: List[Inf] = []
    taken_gts: Set[int] = set()

    gt_objects = ground_truths
    if ignored_ground_truths:
        gt_objects = gt_objects + ignored_ground_truths

    # sort inferences by highest confidence first
    inferences = sorted(inferences, key=lambda inf: -inf.score)

    # for each inference, find the ground truth with the highest IOU
    for inf in inferences:
        best_gt = None
        best_gt_iou = -1.0
        for g, gt in enumerate(gt_objects):
            inf_gt_iou = iou(gt, inf)
            # track the highest iou over the threshold
            if inf_gt_iou >= iou_threshold and inf_gt_iou > best_gt_iou:
                best_gt_iou = inf_gt_iou
                best_gt = g

        if best_gt is None or (best_gt in taken_gts and best_gt < len(ground_truths)):
            # if there are no potential matches, or the best non-ignored gt is already taken, this inf has no match
            unmatched_inf.append(inf)
        elif best_gt < len(ground_truths):
            # if the best non-ignored gt is able to be taken
            matched.append((ground_truths[best_gt], inf))
            taken_gts.add(best_gt)

    unmatched_gt = [gt for gt_idx, gt in enumerate(ground_truths) if gt_idx not in taken_gts]
    return InferenceMatches(matched=matched, unmatched_gt=unmatched_gt, unmatched_inf=unmatched_inf)


def match_inferences(
    ground_truths: List[GT],
    inferences: List[Inf],
    *,
    ignored_ground_truths: Optional[List[GT]] = None,
    mode: Literal["pascal"] = "pascal",
    iou_threshold: float = 0.5,
) -> InferenceMatches[GT, Inf]:
    """
    Matches lists of inferences and ground truths using the provided configuration. This matcher does not consider
    labels, which is appropriate for single class object matching. For matchings with multiple classes, use the
    multiclass matcher.

    PASCAL VOC - For every inference by order of highest confidence, the ground truth of highest IOU is its match.
    Multiple inferences are able to match with the same ignored ground truth.
    See the `PASCAL VOC paper <https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf>`_ for more information.

    :param ground_truths: A list of BoundingBox or a Polygon ground truths.
    :param inferences: A list of scored BoundingBox or a Polygon inferences.
    :param ignored_ground_truths: A list of BoundingBox or a Polygon ground truths to ignore.
    :param mode: The type of matching methodology to use: ``pascal``, more to come...
    :param iou_threshold: The IOU threshold cutoff for valid matchings.
    :return: the matches, unmatched ground truths, and unmatched inferences.
    """

    if mode == "pascal":
        return _match_inferences_single_class_pascal_voc(
            ground_truths,
            inferences,
            ignored_ground_truths=ignored_ground_truths,
            iou_threshold=iou_threshold,
        )

    raise ValueError(f"Mode: '{mode}' is not a valid mode.")
