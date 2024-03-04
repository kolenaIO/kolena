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
from typing import List
from typing import Optional

import numpy as np

from kolena.annotation import BoundingBox
from kolena.annotation import BoundingBox3D

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


def extend_matrix(mat: np.ndarray) -> np.ndarray:
    return np.concatenate([mat, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)


def rect_to_4x4(r0_rect: np.ndarray) -> np.ndarray:
    rect_4x4 = np.zeros([4, 4], dtype=r0_rect.dtype)
    rect_4x4[3, 3] = 1.0
    rect_4x4[:3, :3] = r0_rect
    return rect_4x4


def bbox_to_kitti_format(bbox: BoundingBox) -> np.array:
    return [bbox.top_left[0], bbox.top_left[1], bbox.bottom_right[0], bbox.bottom_right[1]]


def center_to_kitti_format(bbox: BoundingBox3D) -> np.array:
    center = bbox.center
    height = bbox.dimensions[1]
    return [center[0], center[1] + (height / 2.0), center[2]]


def alpha_from_bbox3D(box: BoundingBox3D, box_lidar: BoundingBox3D) -> float:
    return -np.arctan2(-box_lidar.center[1], box_lidar.center[0]) + box.rotations[1]


def transform_to_camera_frame(
    velodyne_bboxes: List[BoundingBox3D],
    transformation: List[float],
    rectification: List[float],
) -> List[BoundingBox3D]:
    def limit_period(val: np.array, offset: Optional[float] = 0.5, period: Optional[float] = np.pi * 2) -> np.ndarray:
        """
        Limit the value into a period for periodic function.
        Args:
            val (np.ndarray): The value to be converted.
            offset (float, optional): Offset to set the value range.
                Defaults to 0.5.
            period (float, optional): Period of the value. Defaults to np.pi*2.
        Returns:
            (np.ndarray): Value in the range of
                [-offset * period, (1-offset) * period]
        """
        limited_val = val - np.floor(val / period + offset) * period  # type: ignore
        return limited_val

    dimensions = [bbox.dimensions for bbox in velodyne_bboxes]
    yaws = limit_period(-np.array([bbox.rotations[2] for bbox in velodyne_bboxes]) - np.pi / 2.0)
    transformation_np = np.array(transformation).reshape([4, 4])
    rectification_np = np.array(rectification).reshape([4, 4])
    centers = np.array([list(bbox.center) + [1.0] for bbox in velodyne_bboxes])
    centers = centers @ (rectification_np @ transformation_np).T
    return [
        BoundingBox3D(
            center=tuple(center[:3]),
            dimensions=(dim[0], dim[2], dim[1]),  # dimension from x, y, z -> x, z, y
            rotations=(0.0, yaw, 0.0),
        )
        for center, dim, yaw in zip(centers, dimensions, yaws)
    ]
