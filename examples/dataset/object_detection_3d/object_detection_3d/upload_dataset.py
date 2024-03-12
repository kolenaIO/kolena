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
from argparse import ArgumentParser
from argparse import Namespace
from collections import Counter

import pandas as pd
from object_detection_3d.constants import BUCKET
from object_detection_3d.constants import DATASET
from object_detection_3d.constants import DEFAULT_DATASET_NAME
from object_detection_3d.constants import ID_FIELDS
from object_detection_3d.constants import TASK

import kolena
from kolena.annotation import LabeledBoundingBox3D
from kolena.asset import ImageAsset
from kolena.dataset import upload_dataset
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.asset import PointCloudAsset

# KITTI only supports evaluation of the first three classes but "Van" and "Person_sitting" GTs
# are used to avoid penalizing inferences labeled as "Car" and "Pedestrian" respectively.
SUPPORTED_LABELS = ["Car", "Pedestrian", "Cyclist", "Van", "Person_sitting", "DontCare"]
LABEL_FILE_COLUMNS = {
    "image_id",
    "left_image",
    "right_image",
    "velodyne",
    "objects",
    "P0",
    "P1",
    "P2",
    "P3",
    "R0_rect",
    "Tr_velo_to_cam",
    "Tr_imu_to_velo",
}


def load_data(df_raw: pd.DataFrame) -> pd.DataFrame:
    records = []
    meta_cols = [col for col in df_raw.columns if col not in LABEL_FILE_COLUMNS]

    for record in df_raw.itertuples():
        bboxes_2d = [
            LabeledBoundingBox(
                label=box["label"],
                top_left=tuple(box["left_top_left"]),
                bottom_right=tuple(box["left_bottom_right"]),
                occluded=box["occluded"],  # type: ignore[call-arg]
                truncated=box["truncated"],  # type: ignore[call-arg]
                alpha=box["alpha"],  # type: ignore[call-arg]
                difficulty=box["difficulty"],  # type: ignore[call-arg]
            )
            for box in record.objects
        ]
        bboxes_3d = [
            LabeledBoundingBox3D(
                label=box["label"],
                dimensions=(box["dim_x"], box["dim_y"], box["dim_z"]),
                center=(box["loc_x"], box["loc_y"], box["loc_z"] + (box["dim_z"] / 2.0)),
                # translate in z (up) axis to be in center
                rotations=(0.0, 0.0, box["rotation_y"]),  # Z yaw axis for lidar coordinates
                occluded=box["occluded"],  # type: ignore[call-arg]
                truncated=box["truncated"],  # type: ignore[call-arg]
                alpha=box["alpha"],  # type: ignore[call-arg]
                difficulty=box["difficulty"],  # type: ignore[call-arg]
            )
            for box in record.objects
        ]
        counts = Counter([r["label"] for r in record.objects])
        metadata = {col: getattr(record, col) for col in meta_cols}
        records.append(
            {
                "image_id": record.image_id,
                "locator": record.left_image,
                "image_bboxes": bboxes_2d,
                "right": ImageAsset(locator=record.right_image),
                "velodyne": PointCloudAsset(locator=record.velodyne),
                "total_objects": len(bboxes_3d),
                "n_car": counts["Car"],
                "n_pedestrian": counts["Pedestrian"],
                "n_cyclist": counts["Cyclist"],
                "velodyne_bboxes": bboxes_3d,
                "velodyne_to_camera_transformation": record.Tr_velo_to_cam,
                "camera_rectification": record.R0_rect,
                "image_projection": record.P2,
                **metadata,
            },
        )

    return pd.DataFrame(records)


def run(args: Namespace) -> int:
    kolena.initialize(verbose=True)
    df_raw = pd.read_json(
        f"s3://{BUCKET}/{DATASET}/{TASK}/raw/{DATASET}.jsonl",
        lines=True,
        orient="records",
        dtype=False,
        storage_options={"anon": True},
    )
    df = load_data(df_raw)
    upload_dataset(args.dataset, df, id_fields=ID_FIELDS)

    return 0


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET_NAME,
        help="Optionally specify a custom dataset name to upload.",
    )
    run(ap.parse_args())
