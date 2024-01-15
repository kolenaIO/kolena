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
from pathlib import Path
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

import numpy as np

from .workflow import GroundTruth
from .workflow import Inference
from .workflow import TestSample
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import BoundingBox3D

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


def get_path_component(component: str, training: bool = True) -> Path:
    if training:
        return Path("training", component)
    else:
        return Path("testing", component)


def get_label_path(datadir: Path) -> Path:
    return datadir / get_path_component("label_2")


def get_calib_path(datadir: Path, training: bool = True) -> Path:
    return datadir / get_path_component("calib", training)


def get_velodyne_path(datadir: Path, training: bool = True) -> Path:
    return datadir / get_path_component("velodyne", training)


# location of PCD data format converted from KITTI velodyne binary data
def get_velodyne_pcd_path(datadir: Path, training: bool = True) -> Path:
    return datadir / get_path_component("velodyne_pcd", training)


def get_left_camera_path(datadir: Path, training: bool = True) -> Path:
    return datadir / get_path_component("image_2", training)


def get_right_camera_path(datadir: Path, training: bool = True) -> Path:
    return datadir / get_path_component("image_3", training)


def get_result_path(datadir: Path, training: bool = True) -> Path:
    return datadir / get_path_component("prediction", training)


def calibration_from_label_id(datadir: Path, label_id: str) -> Dict[str, np.ndarray]:
    def extend_matrix(mat: np.ndarray) -> np.ndarray:
        return np.concatenate([mat, np.array([[0.0, 0.0, 0.0, 1.0]])], axis=0)

    calibration_filepath = get_calib_path(datadir) / f"{label_id}.txt"
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
        limited_val = val - np.floor(val / period + offset) * period
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


def ground_truth_to_kitti_format(sample: TestSample, gt: GroundTruth) -> Dict[str, Any]:
    if len(gt.bboxes_2d) == 0:
        return dict(
            name=np.array([]),
            truncated=np.array([]),
            occluded=np.array([]),
            alpha=np.array([]),
            bbox=np.zeros([0, 4]),
            dimensions=np.zeros([0, 3]),
            location=np.zeros([0, 3]),
            rotation_y=np.array([]),
        )

    camera_bboxes = transform_to_camera_frame(
        gt.bboxes_3d,
        sample.velodyne_to_camera_transformation,
        sample.camera_rectification,
    )
    return dict(
        name=np.array([bbox.label for bbox in gt.bboxes_3d]),
        truncated=np.array([bbox.truncated for bbox in gt.bboxes_3d]),
        occluded=np.array([bbox.occluded for bbox in gt.bboxes_3d]),
        alpha=np.array(
            [alpha_from_bbox3D(bbox, bbox_lidar) for bbox, bbox_lidar in zip(camera_bboxes, gt.bboxes_3d)],
        ),
        bbox=np.array([bbox_to_kitti_format(bbox) for bbox in gt.bboxes_2d]),
        dimensions=np.array([bbox.dimensions for bbox in camera_bboxes]),
        location=np.array([center_to_kitti_format(bbox) for bbox in camera_bboxes]),
        rotation_y=np.array([bbox.rotations[1] for bbox in camera_bboxes]),
    )


def inference_to_kitti_format(sample: TestSample, inf: Inference) -> Dict[str, Any]:
    if len(inf.bboxes_2d) == 0:
        return dict(
            name=np.array([]),
            truncated=np.array([]),
            occluded=np.array([]),
            alpha=np.array([]),
            bbox=np.zeros([0, 4]),
            dimensions=np.zeros([0, 3]),
            location=np.zeros([0, 3]),
            rotation_y=np.array([]),
            score=np.array([]),
        )

    camera_bboxes = transform_to_camera_frame(
        inf.bboxes_3d,
        sample.velodyne_to_camera_transformation,
        sample.camera_rectification,
    )
    return dict(
        name=np.array([bbox.label for bbox in inf.bboxes_3d]),
        truncated=np.array([0.0] * len(inf.bboxes_3d)),
        occluded=np.array([0] * len(inf.bboxes_3d)),
        alpha=np.array(
            [alpha_from_bbox3D(bbox, bbox_lidar) for bbox, bbox_lidar in zip(camera_bboxes, inf.bboxes_3d)],
        ),
        bbox=np.array([bbox_to_kitti_format(bbox) for bbox in inf.bboxes_2d]),
        dimensions=np.array([bbox.dimensions for bbox in camera_bboxes]),
        location=np.array([center_to_kitti_format(bbox) for bbox in camera_bboxes]),
        rotation_y=np.array([bbox.rotations[1] for bbox in camera_bboxes]),
        score=np.array([bbox.score for bbox in inf.bboxes_3d]),
    )
