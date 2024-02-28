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

from kolena.annotation import LabeledBoundingBox


def process_ped_annotations(ped_annotations: Dict[str, Dict[str, Any]]) -> Dict[str, List[LabeledBoundingBox]]:
    bboxes_per_ped = {
        ped_id: [
            LabeledBoundingBox(  # type: ignore
                top_left=(bbox[0], bbox[1]),
                bottom_right=(bbox[2], bbox[3]),
                frame_id=frame_id,
                ped_id=ped_id,
                occlusion=occlusion,
                label="is_crossing" if "b" in ped_id and ped_ann["attributes"]["crossing"] > 0 else "not_crossing",
            )
            for frame_id, bbox, occlusion in zip(ped_ann["frames"], ped_ann["bbox"], ped_ann["occlusion"])
        ]
        for ped_id, ped_ann in ped_annotations.items()
    }
    for ped in bboxes_per_ped:
        bboxes_per_ped[ped].sort(key=lambda x: x.frame_id)  # type: ignore

    filtered_bboxes = {}
    # remove small bboxes
    for ped in bboxes_per_ped:
        bboxes = []
        for bbox in bboxes_per_ped[ped]:
            if bbox.area > 100:
                bboxes.append(bbox)
        filtered_bboxes[ped] = bboxes

    return filtered_bboxes
