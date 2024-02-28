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
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List

from crossing_pedestrian_detection.constants import BUCKET
from crossing_pedestrian_detection.constants import DATASET
from crossing_pedestrian_detection.utils import process_ped_annotations
from smart_open import open as smart_open

from kolena.annotation import ScoredLabeledBoundingBox


def postprocess_inferences(inferences: List[ScoredLabeledBoundingBox]) -> List[ScoredLabeledBoundingBox]:
    n_observation_frames = 16
    in_observation_state = True
    previous_conf = -1.0
    processed_inferences = []
    for i, inf in enumerate(inferences):
        confidence_score = inf.confidence  # type: ignore
        if in_observation_state and i > n_observation_frames:
            in_observation_state = False

        if inf.failed_to_infer:  # type: ignore
            if previous_conf >= 0.0:
                confidence_score = previous_conf
            else:
                confidence_score = 0.0
        else:
            if in_observation_state:
                in_observation_state = False
                if inf.confidence is not None:  # type: ignore
                    previous_conf = inf.confidence  # type: ignore
            else:
                if inf.confidence is not None:  # type: ignore
                    previous_conf = inf.confidence  # type: ignore
        fail_inf = inf.failed_to_infer if not in_observation_state and previous_conf < 0 else False  # type: ignore
        processed_inferences.append(
            ScoredLabeledBoundingBox(  # type: ignore
                ped_id=inf.ped_id,  # type: ignore
                top_left=inf.top_left,  # type: ignore
                bottom_right=inf.bottom_right,  # type: ignore
                frame_id=inf.frame_id,  # type: ignore
                occlusion=inf.occlusion,  # type: ignore
                time_to_event=inf.time_to_event,  # type: ignore
                failed_to_infer=fail_inf,
                confidence=confidence_score,
                observation=in_observation_state,
            ),
        )

    return processed_inferences


def process_data(action_model_name: str, detection_model_name: str) -> Dict[str, List[ScoredLabeledBoundingBox]]:
    model_pkl_name = f"s3://{BUCKET}/{DATASET}/data_cache/jaad_{action_model_name}_{detection_model_name}_database.pkl"
    with smart_open(model_pkl_name, "rb") as inf_file:
        inf_annotations = pickle.load(inf_file)

    results_pkl = f"s3://{BUCKET}/{DATASET}/results/{action_model_name}_{detection_model_name}.pkl"
    with smart_open(results_pkl, "rb") as results_file:
        results = pickle.load(results_file)
        predictions: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for pid, tte, y, img in zip(results["pid"], results["tte"], results["y"], results["image"]):
            pid = pid[0][0]
            frame_id = int(Path(img[0]).stem)
            predictions[pid][frame_id]["tte"] = tte[0]
            predictions[pid][frame_id]["confidence"] = y[0]

    scored_bboxes_per_ped: Dict[str, List[ScoredLabeledBoundingBox]] = {}
    for filename in inf_annotations:
        bboxes_per_ped = process_ped_annotations(inf_annotations[filename]["ped_annotations"])
        scored_bboxes_per_ped[filename] = []
        for ped_id in bboxes_per_ped:
            for bbox in bboxes_per_ped[ped_id]:
                if ped_id not in predictions or bbox.frame_id not in predictions[ped_id]:  # type: ignore
                    failed_to_infer = True
                else:
                    failed_to_infer = False
                    confidence = predictions[ped_id][bbox.frame_id]["confidence"]  # type: ignore
                    time_to_event = predictions[ped_id][bbox.frame_id]["tte"]  # type: ignore
                scored_bboxes_per_ped[filename].append(
                    ScoredLabeledBoundingBox(  # type: ignore
                        ped_id=bbox.ped_id,  # type: ignore
                        top_left=bbox.top_left,  # type: ignore
                        bottom_right=bbox.bottom_right,  # type: ignore
                        frame_id=bbox.frame_id,  # type: ignore
                        occlusion=bbox.occlusion,  # type: ignore
                        time_to_event=time_to_event if not failed_to_infer else None,
                        failed_to_infer=failed_to_infer,
                        confidence=confidence if not failed_to_infer else None,
                    ),
                )
        scored_bboxes_per_ped[filename] = postprocess_inferences(scored_bboxes_per_ped[filename])

    return scored_bboxes_per_ped
