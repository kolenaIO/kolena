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

import os
import pickle
from pathlib import Path

import pandas as pd

import kolena.io
from kolena._experimental.object_detection import upload_object_detection_results
from kolena.annotation import BoundingBox3D
from kolena.annotation import ScoredLabeledBoundingBox

LOCATOR_PREFIX = "s3://kolena-dev-data/kitti/3d-object-detection"
RESULTS_PICKLE_FILE = "hv_second_secfpn_6x8_80e_kitti-3d-3class.pkl"
DATASET_NAME = "kitti-3d-object-detection"
LABEL_OPTIONS = ["Pedestrian", "Cyclist", "Car"]


def load_result(raw: dict, label_options: list[str]) -> dict:
    image_filename = Path(raw["image_path"]).name
    locator = f"{LOCATOR_PREFIX}/training/image_2/{image_filename}"
    raw_inferences = [
        as_2d_bbox(coords, score, label_index, label_options)
        for coords, score, label_index in zip(raw["bbox"], raw["scores"], raw["label_preds"])
    ]
    raw_inferences_3d = [
        as_3d_bbox(coords, score, label_index, label_options)
        for coords, score, label_index in zip(raw["box3d_lidar"], raw["scores"], raw["label_preds"])
    ]
    return dict(
        locator=locator,
        raw_inferences=raw_inferences,
        raw_inferences_3d=raw_inferences_3d,
    )


def as_2d_bbox(
    coords: list[float], score: float, label_index: int, label_options: list[str]
) -> ScoredLabeledBoundingBox:
    return ScoredLabeledBoundingBox(
        label=label_options[label_index],
        score=score,
        top_left=(coords[0], coords[1]),
        bottom_right=(coords[2], coords[3]),
    )


def as_3d_bbox(
    coords: list[float],
    score: float,
    label_index: int,
    label_options: list[str],
) -> BoundingBox3D:
    return BoundingBox3D(
        label=label_options[label_index],
        score=score,
        dimensions=(coords[3], coords[4], coords[5]),
        center=(coords[0], coords[1], coords[2] + (coords[5] / 2.0)),
        rotations=(0.0, 0.0, coords[6]),
    )


def main():
    model_name = Path(RESULTS_PICKLE_FILE).stem
    with open(RESULTS_PICKLE_FILE, "rb") as f:
        raw_results = pickle.load(f)
    df_results = pd.DataFrame([load_result(raw, LABEL_OPTIONS) for raw in raw_results])

    kolena.initialize(api_token=os.environ["KOLENA_TOKEN"], verbose=True)
    upload_object_detection_results(
        DATASET_NAME,
        model_name,
        df_results,
        iou_threshold=0.5,
        threshold_strategy="F1-Optimal",
        min_confidence_score=0.01,
    )


main()
