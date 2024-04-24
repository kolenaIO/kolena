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
from collections import Counter
from typing import Any
from typing import Dict
from typing import List
from typing import NamedTuple
from typing import Optional
from typing import Tuple

import numpy as np
import pandas as pd

from kolena.annotation import BoundingBox
from kolena.annotation import BoundingBox3D
from kolena.annotation import LabeledBoundingBox
from kolena.annotation import LabeledBoundingBox3D
from kolena.asset import ImageAsset
from kolena.asset import PointCloudAsset

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


def kitti_matrix_to_array(arr: List[float], shape: Tuple[int, int]) -> np.array:
    return np.array(arr).reshape(shape)


def create_velo_to_pixel_matrix(velo_to_cam: List[float], cam_to_pixel: List[float]) -> List[List[float]]:
    velo_to_cam_arr = kitti_matrix_to_array(velo_to_cam, (4, 4))
    cam_to_pixel_arr = kitti_matrix_to_array(cam_to_pixel, (4, 4))
    composed_arr = cam_to_pixel_arr @ velo_to_cam_arr  # type: ignore[operator]
    return composed_arr[:3, :].tolist()


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
                "images": [
                    ImageAsset(
                        locator=record.left_image,
                        side="left",  # type: ignore[call-arg]
                        # type: ignore[call-arg]
                        projection=create_velo_to_pixel_matrix(record.Tr_velo_to_cam, record.P2),
                        velodyne_bboxes=bboxes_3d,  # type: ignore[call-arg]
                    ),
                    ImageAsset(
                        locator=record.right_image,
                        side="right",  # type: ignore[call-arg]
                        # type: ignore[call-arg]
                        projection=create_velo_to_pixel_matrix(record.Tr_velo_to_cam, record.P3),
                        velodyne_bboxes=bboxes_3d,  # type: ignore[call-arg]
                    ),
                ],
                "velodyne": PointCloudAsset(locator=record.velodyne),
                "total_objects": len(bboxes_3d),
                "n_car": counts["Car"],
                "n_pedestrian": counts["Pedestrian"],
                "n_cyclist": counts["Cyclist"],
                "velodyne_bboxes": bboxes_3d,
                "velodyne_to_camera_transformation": record.Tr_velo_to_cam,
                "camera_rectification": record.R0_rect,
                "projection": create_velo_to_pixel_matrix(record.Tr_velo_to_cam, record.P2),
                **metadata,
            },
        )

    return pd.DataFrame(records)


def _compute_thresholded_metrics(
    matched_inference: List,
    unmatched_inference: List,
    unmatched_ground_truth: List,
    thresholds: List[float],
    label: str,
) -> List[dict[str, Any]]:
    metrics = []
    for threshold in thresholds:
        count_tp = sum(1 for inf in matched_inference if inf.score >= threshold)
        count_fp = sum(1 for inf in unmatched_inference if inf.score >= threshold)
        count_fn = len(unmatched_ground_truth) + len(matched_inference) - count_tp
        metrics.append(dict(threshold=threshold, label=label, tp=count_tp, fp=count_fp, fn=count_fn))

    return metrics


def _prepare_thresholded_metrics(
    record: NamedTuple,
    raw_result: Dict[str, Dict],
    thresholds: List[float],
    labels: List[str],
) -> List[dict[str, Any]]:
    thresholded_metrics = []
    for label in labels:
        matched_inference = [
            inferences
            for inferences, tp in zip(
                record.raw_inferences_3d,  # type: ignore
                raw_result[label]["tp"],
            )
            if tp
        ]
        unmatched_inference = [
            inferences
            for inferences, fp in zip(
                record.raw_inferences_3d,  # type: ignore
                raw_result[label]["fp"],
            )
            if fp
        ]
        unmatched_ground_truth = [
            velodyne_bboxes
            for velodyne_bboxes, fn in zip(
                record.velodyne_bboxes,  # type: ignore
                raw_result[label]["fn"],
            )
            if fn
        ]
        thresholded_metrics.extend(
            _compute_thresholded_metrics(
                matched_inference,
                unmatched_inference,
                unmatched_ground_truth,
                thresholds,
                label,
            ),
        )

    return thresholded_metrics
