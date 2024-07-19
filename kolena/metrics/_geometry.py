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
import dataclasses
from collections import defaultdict
from typing import Callable
from typing import Dict
from typing import Generic
from typing import List
from typing import Literal
from typing import Optional
from typing import Set
from typing import Tuple
from typing import TypeVar
from typing import Union

from shapely.geometry import Polygon as ShapelyPolygon
from shapely.validation import make_valid

from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena.annotation import BoundingBox
from kolena.annotation import Polygon
from kolena.annotation import ScoredBoundingBox
from kolena.annotation import ScoredLabeledBoundingBox
from kolena.annotation import ScoredLabeledPolygon
from kolena.annotation import ScoredPolygon
from kolena.errors import InputValidationError


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
    Compute the Intersection Over Union (IoU) of two geometries.

    <div class="grid cards" markdown>
    - :kolena-metrics-glossary-16: Metrics Glossary: [Intersection over Union (IoU) ↗](../metrics/iou.md)
    </div>

    :param a: The first geometry in computation.
    :param b: The second geometry in computation.
    :return: The value of the IoU between geometries `a` and `b`.
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


def _inf_with_iou(inf: Inf, iou_val: float) -> Inf:
    exclude = ["iou"] + [f.name for f in dataclasses.fields(inf) if not f.init]
    obj = {k: v for k, v in vars(inf).items() if k not in exclude}
    out: Inf = inf.__class__(**obj, iou=iou_val)  # type: ignore[call-arg,assignment]
    return out


@dataclass(frozen=True)
class InferenceMatches(Generic[GT, Inf]):
    """
    The result of [`match_inferences`][kolena.metrics.match_inferences], providing lists of matches between
    ground truth and inference objects, unmatched ground truths, and unmatched inferences. After applying some
    confidence threshold on returned inference objects, `InferenceMatches` can be used to calculate metrics such as
    precision and recall.

    Objects are of type [`BoundingBox`][kolena.annotation.BoundingBox] or
    [`Polygon`][kolena.annotation.Polygon], depending on the type of inputs provided to
    [`match_inferences`][kolena.metrics.match_inferences].
    """

    matched: List[Tuple[GT, Inf]]
    """
    Pairs of matched ground truth and inference objects above the IoU threshold, along with the calculated IoU.
    Considered as true positive detections after applying some confidence threshold.
    """

    unmatched_gt: List[GT]
    """Unmatched ground truth objects. Considered as false negatives."""

    unmatched_inf: List[Inf]
    """
    Unmatched inference objects, along with the maximum IoU over all ground truths. Considered as false positives
    after applying some confidence threshold.
    """


@dataclass(frozen=True)
class MulticlassInferenceMatches(Generic[GT, Inf]):
    """
    The result of [`match_inferences_multiclass`][kolena.metrics.match_inferences_multiclass], providing lists
    of matches between ground truth and inference objects, unmatched ground truths, and unmatched inferences.

    Unmatched ground truths may be matched with an inference of a different class when no inference of its own class is
    suitable, i.e. a "confused" match. `MultiClassInferenceMatches` can be used to calculate metrics such as precision
    and recall per class, after applying some confidence threshold on the returned inference objects.

    Objects are of type [`LabeledBoundingBox`][kolena.annotation.LabeledBoundingBox] or
    [`LabeledPolygon`][kolena.annotation.LabeledPolygon], depending on the type of inputs provided to
    [`match_inferences_multiclass`][kolena.metrics.match_inferences_multiclass].
    """

    matched: List[Tuple[GT, Inf]]
    """
    Pairs of matched ground truth and inference objects above the IoU threshold, along with the calculated IoU.
    Considered as true positive detections after applying some confidence threshold.
    """

    unmatched_gt: List[Tuple[GT, Optional[Inf]]]
    """
    Pairs of unmatched ground truth objects with its confused inference object (i.e. IoU above threshold with
    mismatching `label`) and calculated IoU, if such an inference exists. Considered as false negatives and
    "confused" detections.
    """

    unmatched_inf: List[Inf]
    """
    Unmatched inference objects, along with the maximum IoU over all ground truths.
    Considered as false positives after applying some confidence threshold.
    """


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

    # for each inference, find the ground truth with the highest IoU
    for inf in inferences:
        match_gt = None
        match_gt_iou = -1.0
        best_gt_iou = 0.0
        for g, gt in enumerate(gt_objects):
            inf_gt_iou = iou(gt, inf)
            # track the highest IoU, regardless of threshold
            if gt not in (ignored_ground_truths or []):
                best_gt_iou = max(best_gt_iou, inf_gt_iou)
            # track the highest IoU over the threshold
            if inf_gt_iou >= iou_threshold and inf_gt_iou > match_gt_iou:
                match_gt_iou = inf_gt_iou
                match_gt = g

        if match_gt is None or (match_gt in taken_gts and match_gt < len(ground_truths)):
            # if there are no potential matches, or the best non-ignored gt is already taken, this inf has no match
            unmatched_inf.append(_inf_with_iou(inf, best_gt_iou))
        elif match_gt < len(ground_truths):
            # if the best non-ignored gt is able to be taken
            matched.append((ground_truths[match_gt], _inf_with_iou(inf, match_gt_iou)))
            taken_gts.add(match_gt)

    unmatched_gt = [gt for gt_idx, gt in enumerate(ground_truths) if gt_idx not in taken_gts]
    return InferenceMatches(matched=matched, unmatched_gt=unmatched_gt, unmatched_inf=unmatched_inf)


def _match_inferences(
    ground_truths: List[GT],
    inferences: List[Inf],
    *,
    ignored_ground_truths: Optional[List[GT]] = None,
    mode: Literal["pascal"] = "pascal",
    iou_threshold: float = 0.5,
) -> InferenceMatches[GT, Inf]:
    if mode == "pascal":
        return _match_inferences_single_class_pascal_voc(
            ground_truths,
            inferences,
            ignored_ground_truths=ignored_ground_truths,
            iou_threshold=iou_threshold,
        )

    raise InputValidationError(f"Mode: '{mode}' is not a valid mode.")


def _get_keyed_items(
    items: List,
    required_match_fields: List[str],
) -> Dict[Tuple, List]:
    keyed_items = defaultdict(list)
    for item in items:
        key = tuple(getattr(item, field, None) for field in required_match_fields)
        keyed_items[key].append(item)
    return keyed_items


def _process_matches(
    keyed_ground_truths: Dict[Tuple, List[GT]],
    keyed_inferences: Dict[Tuple, List[Inf]],
    keyed_ignore_ground_truths: Dict[Tuple, List[GT]],
    match_fn: Callable[..., Union[InferenceMatches, MulticlassInferenceMatches]],
    mode: str,
    iou_threshold: float,
) -> Tuple[List[Tuple[GT, Inf]], List, List[Inf]]:
    keys = {*keyed_inferences.keys(), *keyed_ground_truths.keys(), *keyed_ignore_ground_truths.keys()}
    inf_matches = [
        match_fn(
            keyed_ground_truths[key],
            keyed_inferences[key],
            ignored_ground_truths=keyed_ignore_ground_truths[key],
            mode=mode,
            iou_threshold=iou_threshold,
        )
        for key in keys
    ]
    flattened_matched: List[Tuple[GT, Inf]] = []
    # typing intentionally vague because InferenceMatches and MulticlassInferenceMatches disagree here
    flattened_unmatched_gt: List = []
    flattened_unmatched_inf: List[Inf] = []
    for inf_match in inf_matches:
        flattened_matched.extend(inf_match.matched)
        flattened_unmatched_gt.extend(inf_match.unmatched_gt)
        flattened_unmatched_inf.extend(inf_match.unmatched_inf)

    return flattened_matched, flattened_unmatched_gt, flattened_unmatched_inf


def match_inferences(
    ground_truths: List[GT],
    inferences: List[Inf],
    *,
    ignored_ground_truths: Optional[List[GT]] = None,
    mode: Literal["pascal"] = "pascal",
    iou_threshold: float = 0.5,
    required_match_fields: Optional[List[str]] = None,
) -> InferenceMatches[GT, Inf]:
    """
    Matches model inferences with annotated ground truths using the provided configuration.

    This matcher does not consider labels, which is appropriate for single class object matching. To match with multiple
    classes (i.e. heeding `label` classifications), use the multiclass matcher
    [`match_inferences_multiclass`][kolena.metrics.match_inferences_multiclass].

    Available modes:

    - `pascal` (PASCAL VOC): For every inference by order of highest confidence, the ground truth of highest IoU is
      its match. Multiple inferences are able to match with the same ignored ground truth. See the
      [PASCAL VOC paper](https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf) for more information.

    <div class="grid cards" markdown>
    - :kolena-metrics-glossary-16: Metrics Glossary: [Geometry Matching ↗](../metrics/geometry-matching.md)
    </div>

    :param List[Geometry] ground_truths: A list of [`BoundingBox`][kolena.annotation.BoundingBox] or
        [`Polygon`][kolena.annotation.Polygon] ground truths.
    :param List[ScoredGeometry] inferences: A list of
        [`ScoredBoundingBox`][kolena.annotation.ScoredBoundingBox] or
        [`ScoredPolygon`][kolena.annotation.ScoredPolygon] inferences.
    :param Optional[List[Geometry]] ignored_ground_truths: Optionally specify a list of
        [`BoundingBox`][kolena.annotation.BoundingBox] or [`Polygon`][kolena.annotation.Polygon]
        ground truths to ignore. These ignored ground truths and any inferences matched with them are
        omitted from the returned [`InferenceMatches`][kolena.metrics.InferenceMatches].
    :param mode: The matching methodology to use. See available modes above.
    :param iou_threshold: The IoU (intersection over union, see [`iou`][kolena.metrics.iou]) threshold for
        valid matches.
    :param Optional[List[str]] required_match_fields: Optionally specify a list of fields that must match between
        the inference and ground truth for them to be considered a match.
    :return: [`InferenceMatches`][kolena.metrics.InferenceMatches] containing the matches (true positives),
        unmatched ground truths (false negatives) and unmatched inferences (false positives).
    """

    if required_match_fields is None or len(required_match_fields) == 0:
        return _match_inferences(
            ground_truths,
            inferences,
            ignored_ground_truths=ignored_ground_truths,
            mode=mode,
            iou_threshold=iou_threshold,
        )

    keyed_inferences = _get_keyed_items(inferences, required_match_fields)
    keyed_ground_truths = _get_keyed_items(ground_truths, required_match_fields)
    keyed_ignore_ground_truths = (
        _get_keyed_items(ignored_ground_truths, required_match_fields) if ignored_ground_truths else defaultdict(list)
    )

    matched, unmatched_gt, unmatched_inf = _process_matches(
        keyed_ground_truths,
        keyed_inferences,
        keyed_ignore_ground_truths,
        _match_inferences,
        mode,
        iou_threshold,
    )
    return InferenceMatches(
        matched=matched,
        unmatched_gt=unmatched_gt,
        unmatched_inf=unmatched_inf,
    )


def _match_inferences_multiclass(
    ground_truths: List[GT],
    inferences: List[Inf],
    *,
    ignored_ground_truths: Optional[List[GT]] = None,
    mode: Literal["pascal"] = "pascal",
    iou_threshold: float = 0.5,
) -> MulticlassInferenceMatches[GT, Inf]:
    matched: List[Tuple[GT, Inf]] = []
    unmatched_gt: List[GT] = []
    unmatched_inf: List[Inf] = []
    gts_by_class: Dict[str, List[GT]] = defaultdict(list)
    infs_by_class: Dict[str, List[Inf]] = defaultdict(list)
    ignored_gts_by_class: Dict[str, List[GT]] = defaultdict(list)
    all_labels: Set[str] = set()

    if mode == "pascal":
        matching_function = _match_inferences_single_class_pascal_voc
    else:
        raise InputValidationError(f"Mode: '{mode}' is not a valid mode.")

    # collect all unique labels, store gts and infs of the same label together
    for gt in ground_truths:
        assert hasattr(gt, "label")
        gts_by_class[gt.label].append(gt)
        all_labels.add(gt.label)

    for inf in inferences:
        assert hasattr(inf, "label")
        infs_by_class[inf.label].append(inf)
        all_labels.add(inf.label)

    if ignored_ground_truths:
        for ignored_gt in ignored_ground_truths:
            assert hasattr(ignored_gt, "label")
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
        assert hasattr(gt, "label") and hasattr(inf, "label")
        if gt.label != inf.label:
            confused.append((gt, inf))
            unmatched_gt.remove(gt)

    return MulticlassInferenceMatches(
        matched=matched,
        unmatched_gt=confused + [(gt, None) for gt in unmatched_gt],
        unmatched_inf=unmatched_inf,
    )


def match_inferences_multiclass(
    ground_truths: List[GT],
    inferences: List[Inf],
    *,
    ignored_ground_truths: Optional[List[GT]] = None,
    mode: Literal["pascal"] = "pascal",
    iou_threshold: float = 0.5,
    required_match_fields: Optional[List[str]] = None,
) -> MulticlassInferenceMatches[GT, Inf]:
    """
    Matches model inferences with annotated ground truths using the provided configuration.

    This matcher considers `label` values matching per class. After matching inferences and ground truths with
    equivalent `label` values, unmatched inferences and unmatched ground truths are matched once more to identify
    confused matches, where localization succeeded (i.e. IoU above `iou_threshold`) but classification failed (i.e.
    mismatching `label` values).

    Available modes:

    - `pascal` (PASCAL VOC): For every inference by order of highest confidence, the ground truth of highest IoU is
      its match. Multiple inferences are able to match with the same ignored ground truth. See the
      [PASCAL VOC paper](https://homepages.inf.ed.ac.uk/ckiw/postscript/ijcv_voc09.pdf) for more information.

    <div class="grid cards" markdown>
    - :kolena-metrics-glossary-16: Metrics Glossary: [Geometry Matching ↗](../metrics/geometry-matching.md)
    </div>

    :param List[LabeledGeometry] ground_truths: A list of
        [`LabeledBoundingBox`][kolena.annotation.LabeledBoundingBox] or
        [`LabeledPolygon`][kolena.annotation.LabeledPolygon] ground truths.
    :param List[ScoredLabeledGeometry] inferences: A list of
        [`ScoredLabeledBoundingBox`][kolena.annotation.ScoredLabeledBoundingBox] or
        [`ScoredLabeledPolygon`][kolena.annotation.ScoredLabeledPolygon] inferences.
    :param Optional[List[LabeledGeometry]] ignored_ground_truths: Optionally specify a list of
        [`LabeledBoundingBox`][kolena.annotation.LabeledBoundingBox] or
        [`LabeledPolygon`][kolena.annotation.LabeledPolygon] ground truths to ignore. These ignored ground
        truths and any inferences matched with them are omitted from the returned
        [`MulticlassInferenceMatches`][kolena.metrics.MulticlassInferenceMatches].
    :param mode: The matching methodology to use. See available modes above.
    :param iou_threshold: The IoU threshold cutoff for valid matches.
    :param Optional[List[str]] required_match_fields: Optionally specify a list of fields that must match between
        the inference and ground truth for them to be considered a match.
    :return:
        [`MulticlassInferenceMatches`][kolena.metrics.MulticlassInferenceMatches] containing the matches
        (true positives), unmatched ground truths (false negatives), and unmatched inferences (false positives).
    """

    if not required_match_fields:
        return _match_inferences_multiclass(
            ground_truths=ground_truths,
            inferences=inferences,
            ignored_ground_truths=ignored_ground_truths,
            mode=mode,
            iou_threshold=iou_threshold,
        )

    keyed_inferences = _get_keyed_items(inferences, required_match_fields)
    keyed_ground_truths = _get_keyed_items(ground_truths, required_match_fields)
    keyed_ignore_ground_truths = (
        _get_keyed_items(ignored_ground_truths, required_match_fields) if ignored_ground_truths else defaultdict(list)
    )

    matched, unmatched_gt, unmatched_inf = _process_matches(
        keyed_ground_truths,
        keyed_inferences,
        keyed_ignore_ground_truths,
        _match_inferences_multiclass,
        mode,
        iou_threshold,
    )

    return MulticlassInferenceMatches(
        matched=matched,
        unmatched_gt=unmatched_gt,
        unmatched_inf=unmatched_inf,
    )
