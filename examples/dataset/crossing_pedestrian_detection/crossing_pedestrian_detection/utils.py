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

from pydantic.v1.dataclasses import dataclass

from kolena.annotation import LabeledBoundingBox
from kolena.annotation import ScoredLabeledBoundingBox

HIGH_RISK_THRESHOLD = 0.9


@dataclass(frozen=True)
class PedestrianBoundingBox(LabeledBoundingBox):
    frame_id: int
    ped_id: str
    occlusion: str
    risk: Optional[str] = None

    def set_risk(self, risk: str) -> None:
        object.__setattr__(self, "risk", risk)


@dataclass(frozen=True)
class ScoredPedestrianBoundingBox(ScoredLabeledBoundingBox):
    frame_id: int
    ped_id: str
    occlusion: str
    time_to_event: Optional[float]
    failed_to_infer: bool
    observation: Optional[bool] = None


@dataclass(frozen=True)
class ProccessedGroundTruth:
    bboxes: List[PedestrianBoundingBox]
    pids: List[str]


def compute_collision_risk(ground_truth_boxes: List[PedestrianBoundingBox]) -> float:
    """Estimates the collision risk by computing the maximum area of the focus pedestrian's bboxes
    and assign 0-1 score. 0 being no risk (area == 1000) and 1 being high risk (area == 45000)."""
    high_risk_bbox_area: float = 45000.0
    low_risk_bbox_area: float = 1000.0
    max_area = max(bbox.area for bbox in ground_truth_boxes)
    score = min(max(0.0, float(max_area - low_risk_bbox_area) / (high_risk_bbox_area - low_risk_bbox_area)), 1.0)
    return score


def process_gt_bboxes(ped_annotations: Dict[str, Dict[str, Any]]) -> ProccessedGroundTruth:
    bboxes_per_ped = process_ped_annotations(ped_annotations)
    bboxes = []
    pids = []

    for ped_id in bboxes_per_ped:
        risk_score = compute_collision_risk(bboxes_per_ped[ped_id])
        ped_bboxes = bboxes_per_ped[ped_id]
        risk_level = "high" if risk_score > HIGH_RISK_THRESHOLD else "low"
        for bbox in ped_bboxes:
            bbox.set_risk(risk_level)
        bboxes.extend(bboxes_per_ped[ped_id])
        pids.append(ped_id)

    return ProccessedGroundTruth(
        bboxes=bboxes,
        pids=pids,
    )


def process_ped_annotations(ped_annotations: Dict[str, Dict[str, Any]]) -> Dict[str, List[PedestrianBoundingBox]]:
    bboxes_per_ped = {}
    for ped_id, ped_ann in ped_annotations.items():
        bboxes = []
        for frame_id, bbox, occlusion in zip(ped_ann["frames"], ped_ann["bbox"], ped_ann["occlusion"]):
            bbox = PedestrianBoundingBox(
                top_left=(bbox[0], bbox[1]),
                bottom_right=(bbox[2], bbox[3]),
                frame_id=frame_id,
                ped_id=ped_id,
                occlusion=str(occlusion),
                label="is_crossing" if "b" in ped_id and ped_ann["attributes"]["crossing"] > 0 else "not_crossing",
            )
            if bbox.area > 100:
                bboxes.append(bbox)
        bboxes_per_ped[ped_id] = bboxes
    return bboxes_per_ped
