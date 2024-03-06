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

from collections import defaultdict
from glob import glob
from pathlib import Path

import pandas as pd
from utils import calibrate_velo_to_cam
from utils import calibration_from_label_id

import kolena.io
from kolena.annotation import BoundingBox
from kolena.annotation import BoundingBox3D
from kolena.asset import ImageAsset
from kolena.asset import PointCloudAsset


def process_metadata_txt(filepath):
    label_id = Path(filepath).stem
    locator = f"s3://kolena-dev-data/kitti/3d-object-detection/training/image_2/{label_id}.png"
    right_locator = f"s3://kolena-dev-data/kitti/3d-object-detection/training/image_3/{label_id}.png"
    pcd_locator = f"s3://kolena-dev-data/kitti/3d-object-detection/training/velodyne_pcd/{label_id}.pcd"
    calibration = calibration_from_label_id(label_id)
    df = calibrate_velo_to_cam(filepath, calibration)

    bboxes_2d = []
    bboxes_3d = []
    counts_by_label = defaultdict(int)
    for row in df.itertuples():
        bbox_2d = BoundingBox(
            label=row.type,
            top_left=(row.bbox_x0, row.bbox_y0),
            bottom_right=(row.bbox_x1, row.bbox_y1),
        )
        bbox_3d = BoundingBox3D(
            label=row.type,
            dimensions=(row.dim_x, row.dim_y, row.dim_z),
            center=(row.loc_x, row.loc_y, row.loc_z + (row.dim_z / 2.0)),  # translate in z (up) axis to be in center
            rotations=(0.0, 0.0, row.rotation_y),  # Z yaw axis for lidar coordinates
            truncated=row.truncated,
            occluded=row.occluded,
            alpha=row.alpha,
        )
        counts_by_label[row.type] += 1
        bboxes_2d.append(bbox_2d)
        bboxes_3d.append(bbox_3d)

    return dict(
        locator=locator,
        right=ImageAsset(locator=right_locator),
        velodyne=PointCloudAsset(locator=pcd_locator),
        total_objects=len(df),
        n_car=counts_by_label["Car"],
        n_pedestrian=counts_by_label["Pedestrian"],
        n_cyclist=counts_by_label["Cyclist"],
        ground_truths=bboxes_2d,
        ground_truths_3d=bboxes_3d,
    )


def main():
    filepaths = glob("data/training/label_2/*.txt")
    df_dataset = pd.DataFrame([process_metadata_txt(filepath) for filepath in filepaths])
    kolena.io.dataframe_to_csv(df_dataset, "kitti-3d-object-detection.csv", index=False)


main()
