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

import numpy as np
import pandas as pd
from mmdet3d.structures import Box3DMode


LABEL_FILE_COLUMNS = [
    "type",
    "truncated",
    "occluded",
    "alpha",
    "bbox_x0",
    "bbox_y0",
    "bbox_x1",
    "bbox_y1",
    "dim_y",
    "dim_z",
    "dim_x",
    "loc_x",
    "loc_y",
    "loc_z",
    "rotation_y",
]


def calibration_from_label_id(label_id: str) -> dict[str, np.ndarray]:
    def extend_matrix(mat: np.ndarray) -> np.ndarray:
        return np.concatenate([mat, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    calibration_filepath = f"data/training/calib/{label_id}.txt"
    with open(calibration_filepath) as f:
        lines = f.readlines()

    p2 = extend_matrix(np.array([float(info) for info in lines[2].split(" ")[1:13]]).reshape([3, 4]))
    r0_rect = np.array([float(info) for info in lines[4].split(" ")[1:10]]).reshape([3, 3])
    rect_4x4 = np.zeros([4, 4], dtype=r0_rect.dtype)
    rect_4x4[3, 3] = 1.0
    rect_4x4[:3, :3] = r0_rect
    velo_to_cam = extend_matrix(np.array([float(info) for info in lines[5].split(" ")[1:13]]).reshape([3, 4]))

    return dict(
        velodyne_to_camera=velo_to_cam,
        camera_rectification=rect_4x4,
        image_projection=p2,
    )


def calibrate_velo_to_cam(label_filepath: str, calibration: dict[str, np.ndarray]) -> pd.DataFrame:
    """Convert KITTI 3D bbox coordinates to image coordinates"""

    df = pd.read_csv(label_filepath, delimiter=" ", header=None, names=LABEL_FILE_COLUMNS)
    camera_bboxes = [
        [record.loc_x, record.loc_y, record.loc_z, record.dim_x, record.dim_y, record.dim_z, record.rotation_y]
        for record in df.itertuples()
    ]

    lidar_bboxes = Box3DMode.convert(
        np.array(camera_bboxes),
        Box3DMode.CAM,
        Box3DMode.LIDAR,
        rt_mat=np.linalg.inv(calibration["camera_rectification"] @ calibration["velodyne_to_camera"]),
        with_yaw=True,
    )

    df["loc_x"] = [bbox[0] for bbox in lidar_bboxes]
    df["loc_y"] = [bbox[1] for bbox in lidar_bboxes]
    df["loc_z"] = [bbox[2] for bbox in lidar_bboxes]
    df["dim_x"] = [bbox[3] for bbox in lidar_bboxes]
    df["dim_y"] = [bbox[4] for bbox in lidar_bboxes]
    df["dim_z"] = [bbox[5] for bbox in lidar_bboxes]
    df["rotation_y"] = [bbox[6] for bbox in lidar_bboxes]

    return df
