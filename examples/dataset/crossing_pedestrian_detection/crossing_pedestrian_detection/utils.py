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
from typing import List
from typing import Optional
from typing import Tuple
from kolena.metrics import iou


from kolena.annotation import LabeledBoundingBox
from kolena.annotation import ScoredLabeledBoundingBox
from crossing_pedestrian_detection.constants import IOU
from pydantic.dataclasses import dataclass

HIGH_RISK_THRESHOLD = 0.9

def compute_collision_risk(ground_truth_boxes: List[LabeledBoundingBox]) -> float:
    """Estimates the collision risk by computing the maximum area of the focus pedestrian's bboxes
    and assign 0-1 score. 0 being no risk (area == 1000) and 1 being high risk (area == 45000)."""
    high_risk_bbox_area: float = 45000.0
    low_risk_bbox_area: float = 1000.0
    max_area = max(bbox.area for bbox in ground_truth_boxes)
    score = min(max(0.0, float(max_area - low_risk_bbox_area) / (high_risk_bbox_area - low_risk_bbox_area)), 1.0)
    return score


def process_gt_bboxes(ped_annotations: Dict[str, Dict[str, Any]]):
    bboxes_per_ped = process_ped_annotations(ped_annotations)
    high_risk_bboxes = []
    low_risk_bboxes = []
    high_risk_pids = []
    low_risk_pids = []

    for ped_id in bboxes_per_ped:
        risk_score = compute_collision_risk(bboxes_per_ped[ped_id])
        if risk_score > HIGH_RISK_THRESHOLD:
            high_risk_bboxes.extend(bboxes_per_ped[ped_id])
            high_risk_pids.append(ped_id)
        else:
            low_risk_bboxes.extend(bboxes_per_ped[ped_id])
            low_risk_pids.append(ped_id)

    return [high_risk_bboxes, low_risk_bboxes], [high_risk_pids, low_risk_pids]


def process_ped_annotations(ped_annotations: Dict[str, Dict[str, Any]]):
    bboxes_per_ped = {}
    for ped_id, ped_ann in ped_annotations.items():
        bboxes = []
        for frame_id, bbox, occlusion in zip(ped_ann["frames"], ped_ann["bbox"], ped_ann["occlusion"]):
            bbox = LabeledBoundingBox(  # type: ignore
                top_left=(bbox[0], bbox[1]),
                bottom_right=(bbox[2], bbox[3]),
                frame_id=frame_id,
                ped_id=ped_id,
                occlusion=occlusion,
                label="is_crossing" if "b" in ped_id and ped_ann["attributes"]["crossing"] > 0 else "not_crossing",
            )
            if bbox.area > 100:
                bboxes.append(bbox)
        bboxes_per_ped[ped_id] = bboxes
    return bboxes_per_ped


@dataclass(frozen=True)
class FrameMatch:
    frame_id: int
    unmatched_gt: Any
    unmatched_inf: Any
    matched: Any
    gt_label: Optional[str]
    inf_label: Optional[str]
    gt: Optional[LabeledBoundingBox]
    iou_threshold: float
    matched_pedestrian: Optional[ScoredLabeledBoundingBox] = None




@dataclass(frozen=True)
class BoundingBoxMatch:
    matched: Optional[
        Tuple[LabeledBoundingBox, ScoredLabeledBoundingBox]
    ] = None  # i.e. true positives, if confidence > threshold
    matched_iou: float = 0.0  # iou for each set of matches

    def is_matched(self) -> bool:
        return self.matched is not None


def create_labeled_bbox(bbox):
    return LabeledBoundingBox(top_left=bbox.top_left,
                              bottom_right=bbox.bottom_right,
                              label=bbox.label,
                              frame_id=bbox.frame_id,
                              ped_id=bbox.ped_id,
                              occlusion=bbox.occlusion)


def match_bounding_boxes(
    gt_bbox: LabeledBoundingBox,
    inf_bboxes: List[ScoredLabeledBoundingBox],
) -> BoundingBoxMatch:
    """Match inferences to ground truths on a given image using the provided configuration."""
    matched_inf: Optional[ScoredLabeledBoundingBox] = None
    matched_iou: float = 0.0

    # Find the inference with the highest IOU (don't have the detection confidence yet)
    best_iou = 0.0
    for inf in inf_bboxes:
        iou_value = iou(gt_bbox, inf)
        if iou_value < 0.5:
            continue

        if iou_value > best_iou:
            matched_inf = inf
            best_iou = iou_value
            matched_iou = iou_value

    matches = BoundingBoxMatch(
        matched=(create_labeled_bbox(gt_bbox), matched_inf) if matched_inf is not None else None,
        matched_iou=matched_iou,
    )

    return matches