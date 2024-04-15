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

import pandas as pd
from object_detection_3d.constants import BUCKET
from object_detection_3d.constants import DATASET
from object_detection_3d.constants import DEFAULT_DATASET_NAME
from object_detection_3d.constants import ID_FIELDS
from object_detection_3d.constants import TASK
from object_detection_3d.utils import load_data

from kolena.dataset import upload_dataset

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


def run(args: Namespace) -> int:
    df_raw = pd.read_json(
        f"s3://{BUCKET}/{DATASET}/{TASK}/raw/{DATASET}.jsonl",
        lines=True,
        orient="records",
        dtype=False,
        storage_options={"anon": True},
    )
    df = load_data(df_raw)
    df.drop(columns=["image_bboxes", "images"], inplace=True)
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
