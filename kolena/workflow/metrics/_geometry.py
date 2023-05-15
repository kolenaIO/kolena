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
from dataclasses import dataclass
from typing import Dict
from typing import Generic
from typing import List
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

from kolena.errors import InputValidationError

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
    :return: the value of the IOU between geometries ``a`` and ``b``.
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


GT = TypeVar("GT", bound=Union[BoundingBox, Polygon])
Inf = TypeVar("Inf", bound=Union[ScoredBoundingBox, ScoredPolygon, ScoredLabeledBoundingBox, ScoredLabeledPolygon])


@dataclass(frozen=True)
class InferenceMatches(Generic[GT, Inf]):
    """
    The result of :func:`match_inferences`, providing lists of matches between ground truth and inference objects,
    unmatched ground truths, and unmatched inferences. After applying some confidence threshold on returned inference
    objects, :class:`InferenceMatches` can be used to calculate metrics such as precision and recall.

    Objects are of type :class:`BoundingBox` or :class:`Polygon`, depending on the type of inputs provided to
    :func:`match_inferences`.
    """

    #: Pairs of matched ground truth and inference objects above the IOU threshold. Considered as true positive
    #: detections after applying some confidence threshold.
    matched: List[Tuple[GT, Inf]]

    #: Unmatched ground truth objects. Considered as false negatives.
    unmatched_gt: List[GT]

    #: Unmatched inference objects. Considered as false positives after applying some confidence threshold.
    unmatched_inf: List[Inf]


def _match_inferences_single_class_pascal_voc(
    ground_truths: List[GT],
    inferences: List[Inf],
    ignored_ground_truths: Optional[List[GT]] = None,
    iou_threshold: float = 0.5,
) -> InferenceMatches[GT, Inf]:
    matched: List[Tuple[GT, Inf]] = []
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
    Matches model inferences with annotated ground truths using the provided configuration.

    This matcher does not consider labels, which is appropriate for single class object matching. To match with multiple
    classes (i.e. heeding ``label`` classifications), use the multiclass matcher :func:`match_inferences_multiclass`.

    Available modes:

    - ``pascal`` (PASCAL VOC): For every inference by order of highest confidence, the ground truth of highest IOU is
      its match. Multiple inferences are able to match with the same ignored ground truth. See the
      `PASCAL VOC paper <https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf>`_ for more information.

    :param List[Geometry] ground_truths: a list of :class:`BoundingBox` or :class:`Polygon` ground truths.
    :param List[ScoredGeometry] inferences: a list of :class:`ScoredBoundingBox` or :class:`ScoredPolygon` inferences.
    :param Optional[List[Geometry]] ignored_ground_truths: optionally specify a list of :class:`BoundingBox` or
        :class:`Polygon` ground truths to ignore. These ignored ground truths and any inferences matched with them are
        omitted from the returned :class:`InferenceMatches`.
    :param Literal["pascal"] mode: the type of matching methodology to use. See available modes above.
    :param iou_threshold: the IOU (intersection over union, see :meth:`iou`) threshold for valid matches.
    :return: :class:`InferenceMatches` containing the matches (true positives), unmatched ground truths (false
        negatives) and unmatched inferences (false positives).
    """

    if mode == "pascal":
        return _match_inferences_single_class_pascal_voc(
            ground_truths,
            inferences,
            ignored_ground_truths=ignored_ground_truths,
            iou_threshold=iou_threshold,
        )

    raise InputValidationError(f"Mode: '{mode}' is not a valid mode.")


GT_Multiclass = TypeVar("GT_Multiclass", bound=Union[LabeledBoundingBox, LabeledPolygon])
Inf_Multiclass = TypeVar("Inf_Multiclass", bound=Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon])


@dataclass(frozen=True)
class MulticlassInferenceMatches(Generic[GT_Multiclass, Inf_Multiclass]):
    """
    The result of :func:`match_inferences_multiclass`, providing lists of matches between ground truth and inference
    objects, unmatched ground truths, and unmatched inferences. The unmatched ground truths may be matched with an
    inference of a different class when no inference of its own class is suitable, a confused match.
    :class:`MultiClassInferenceMatches` can be used to calculate metrics such as precision and recall per class, after
    applying some confidence threshold on the returned inference objects.

    Objects are of type :class:`BoundingBox` or :class:`Polygon`, depending on the type of inputs provided to
    :func:`match_inferences`.
    """

    #: Pairs of matched ground truth and inference objects above the IOU threshold. Considered as true positive
    #: detections after applying some confidence threshold.
    matched: List[Tuple[GT_Multiclass, Inf_Multiclass]]

    #: Pairs of unmatched ground truth objects with its confused inference object (i.e. IOU above threshold with
    # mismatching ``label``), if such an inference exists. Considered as false negatives and "confused" detections.
    unmatched_gt: List[Tuple[GT_Multiclass, Optional[Inf_Multiclass]]]

    #: Unmatched inference objects. Considered as false positives after applying some confidence threshold.
    unmatched_inf: List[Inf_Multiclass]


def match_inferences_multiclass(
    ground_truths: List[GT_Multiclass],
    inferences: List[Inf_Multiclass],
    *,
    ignored_ground_truths: Optional[List[GT_Multiclass]] = None,
    mode: Literal["pascal"] = "pascal",
    iou_threshold: float = 0.5,
) -> MulticlassInferenceMatches[GT_Multiclass, Inf_Multiclass]:
    """
    Matches model inferences with annotated ground truths using the provided configuration.

    This matcher considers ``label`` values matching per class. After matching inferences and ground truths with
    equivalent ``label`` values, unmatched inferences and unmatched ground truths are matched once more to identify
    confused matches, where localization succeeded (i.e. IOU above ``iou_threshold``) but classification failed (i.e.
    mismatching ``label`` values).

    Available modes:

    - ``pascal`` (PASCAL VOC): For every inference by order of highest confidence, the ground truth of highest IOU is
      its match. Multiple inferences are able to match with the same ignored ground truth. See the
      `PASCAL VOC paper <https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf>`_ for more information.

    :param List[LabeledGeometry] ground_truths: a list of :class:`LabeledBoundingBox` or :class:`LabeledPolygon` ground
        truths.
    :param List[ScoredLabeledGeometry] inferences: a list of :class:`ScoredLabeledBoundingBox` or
        :class:`ScoredLabeledPolygon` inferences.
    :param Optional[List[LabeledGeometry]] ignored_ground_truths: optionally specify a list of
        :class:`LabeledBoundingBox` or :class:`LabeledPolygon` ground truths to ignore. These ignored ground truths any
        any inferences matched with them are omitted from the returned :class:`MulticlassInferenceMatches`.
    :param Literal["pascal"] mode: The type of matching methodology to use. See available modes above.
    :param iou_threshold: The IOU threshold cutoff for valid matches.
    :return: :class:`MulticlassInferenceMatches` containing the matches (true positives), unmatched ground truths (false
        negatives), and unmatched inferences (false positives).
    """
    matched: List[Tuple[GT_Multiclass, Inf_Multiclass]] = []
    unmatched_gt: List[GT_Multiclass] = []
    unmatched_inf: List[Inf_Multiclass] = []
    gts_by_class: Dict[str, List[GT_Multiclass]] = defaultdict(list)
    infs_by_class: Dict[str, List[Inf_Multiclass]] = defaultdict(list)
    ignored_gts_by_class: Dict[str, List[GT_Multiclass]] = defaultdict(list)
    all_labels: Set[str] = set()

    if mode == "pascal":
        matching_function = _match_inferences_single_class_pascal_voc
    else:
        raise InputValidationError(f"Mode: '{mode}' is not a valid mode.")

    # collect all unique labels, store gts and infs of the same label together
    for gt in ground_truths:
        gts_by_class[gt.label].append(gt)
        all_labels.add(gt.label)

    for inf in inferences:
        infs_by_class[inf.label].append(inf)
        all_labels.add(inf.label)

    if ignored_ground_truths:
        for ignored_gt in ignored_ground_truths:
            ignored_gts_by_class[ignored_gt.label].append(ignored_gt)

    for label in sorted(all_labels):
        ground_truths_single = gts_by_class[label]
        inferences_single = infs_by_class[label]
        ignored_ground_truths_single = ignored_gts_by_class[label]

        single_matches: InferenceMatches = matching_function(
            ground_truths_single,
            inferences_single,
            ignored_ground_truths=ignored_ground_truths_single,
            iou_threshold=iou_threshold,
        )

        matched += single_matches.matched
        unmatched_gt += single_matches.unmatched_gt
        unmatched_inf += single_matches.unmatched_inf

    confused_matches = matching_function(
        unmatched_gt,
        unmatched_inf,
        ignored_ground_truths=ignored_ground_truths,
        iou_threshold=iou_threshold,
    )

    confused = []
    for gt, inf in confused_matches.matched:
        if gt.label != inf.label:
            confused.append((gt, inf))
            unmatched_gt.remove(gt)

    return MulticlassInferenceMatches(
        matched=matched,
        unmatched_gt=confused + [(gt, None) for gt in unmatched_gt],
        unmatched_inf=unmatched_inf,
    )
