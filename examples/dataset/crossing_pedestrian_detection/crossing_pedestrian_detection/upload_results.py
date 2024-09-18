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
from argparse import ArgumentParser
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List

import boto3
import botocore
import pandas as pd
from botocore.client import Config
from crossing_pedestrian_detection.constants import BUCKET
from crossing_pedestrian_detection.constants import DATASET
from crossing_pedestrian_detection.constants import DEFAULT_DATASET_NAME
from crossing_pedestrian_detection.constants import MODELS
from crossing_pedestrian_detection.utils import process_ped_annotations
from crossing_pedestrian_detection.utils import ScoredPedestrianBoundingBox
from smart_open import open as smart_open

from kolena._experimental.object_detection import upload_object_detection_results

TRANSPORT_PARAMS = {"client": boto3.client("s3", config=Config(signature_version=botocore.UNSIGNED))}

THRESHOLD = 0.5


def postprocess_inferences(inferences: List[ScoredPedestrianBoundingBox]) -> List[ScoredPedestrianBoundingBox]:
    n_observation_frames = 16
    in_observation_state = True
    previous_conf = -1.0
    processed_inferences = []
    for i, inf in enumerate(inferences):
        confidence_score = inf.score
        if in_observation_state and i > n_observation_frames:
            in_observation_state = False

        if inf.failed_to_infer:
            if previous_conf >= 0.0:
                confidence_score = previous_conf
            else:
                confidence_score = 0.0
        else:
            if in_observation_state:
                in_observation_state = False
                if inf.score is not None:
                    previous_conf = inf.score
            else:
                if inf.score is not None:
                    previous_conf = inf.score
        fail_inf = inf.failed_to_infer if not in_observation_state and previous_conf < 0 else False
        processed_inferences.append(
            ScoredPedestrianBoundingBox(
                ped_id=inf.ped_id,
                top_left=inf.top_left,
                bottom_right=inf.bottom_right,
                frame_id=inf.frame_id,
                occlusion=inf.occlusion,
                time_to_event=inf.time_to_event,
                failed_to_infer=fail_inf,
                score=confidence_score,
                observation=in_observation_state,
                label=inf.label,
            ),
        )

    return processed_inferences


def process_inf_data(action_model_name: str, detection_model_name: str) -> Dict[str, List[ScoredPedestrianBoundingBox]]:
    model_pkl_name = f"s3://{BUCKET}/{DATASET}/raw/jaad_{action_model_name}_{detection_model_name}_database.pkl"
    # access S3 anonymously
    # adapted from https://github.com/piskvorky/smart_open/blob/develop/howto.md#how-to-access-s3-anonymously
    with smart_open(model_pkl_name, "rb", transport_params=TRANSPORT_PARAMS) as inf_file:
        inf_annotations = pickle.load(inf_file)

    results_pkl = f"s3://{BUCKET}/{DATASET}/results/raw/{action_model_name}_{detection_model_name}.pkl"
    with smart_open(results_pkl, "rb", transport_params=TRANSPORT_PARAMS) as results_file:
        results = pickle.load(results_file)
        predictions: Dict[str, Dict[int, Dict[str, float]]] = defaultdict(lambda: defaultdict(dict))
        for pid, tte, y, img in zip(results["pid"], results["tte"], results["y"], results["image"]):
            pid = pid[0][0]
            frame_id = int(Path(img[0]).stem)
            predictions[pid][frame_id]["tte"] = tte[0]
            predictions[pid][frame_id]["confidence"] = y[0]

    scored_bboxes_per_ped: Dict[str, List[ScoredPedestrianBoundingBox]] = {}
    for filename in inf_annotations:
        bboxes_per_file = process_ped_annotations(inf_annotations[filename]["ped_annotations"])
        scored_bboxes_lst = []
        for ped_id, bboxes in bboxes_per_file.items():
            for bbox in bboxes:
                if bbox.ped_id not in predictions or bbox.frame_id not in predictions[ped_id]:
                    failed_to_infer = True
                else:
                    failed_to_infer = False
                    confidence = predictions[bbox.ped_id][bbox.frame_id]["confidence"]
                    time_to_event = predictions[bbox.ped_id][bbox.frame_id]["tte"]

                scored_bboxes_lst.append(
                    ScoredPedestrianBoundingBox(
                        ped_id=bbox.ped_id,
                        top_left=bbox.top_left,
                        bottom_right=bbox.bottom_right,
                        frame_id=bbox.frame_id,
                        occlusion=bbox.occlusion,
                        time_to_event=time_to_event if not failed_to_infer else None,
                        failed_to_infer=failed_to_infer,
                        score=confidence if not failed_to_infer else 0,
                        label=bbox.label,
                    ),
                )
        locator = f"s3://kolena-public-examples/JAAD/data/videos/{filename}.mp4"
        scored_bboxes_per_ped[locator] = postprocess_inferences(scored_bboxes_lst)

    return scored_bboxes_per_ped


def run(args: Namespace) -> None:
    split_model = args.model.split("_")
    inference_data = process_inf_data(split_model[0], split_model[1])
    df_inf = pd.DataFrame(list(inference_data.items()), columns=["locator", "raw_inferences"])

    if args.sample_count > 0:
        df_inf = df_inf[: args.sample_count]

    upload_object_detection_results(
        args.dataset,
        args.model,
        df_inf,
        required_match_fields=["frame_id"],
        threshold_strategy=THRESHOLD,
    )


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=MODELS, help="Name of the model to test.")
    ap.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Optionally specify a custom dataset name to test.",
    )
    ap.add_argument(
        "--sample-count",
        type=int,
        default=0,
        help="Number of samples to use. All samples are used by default.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
