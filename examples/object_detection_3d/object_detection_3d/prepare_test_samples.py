# Copyright 2021-2023 Kolena Inc.
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
import dataclasses
import glob
import json
import math
import os
import posixpath
from argparse import ArgumentParser
from argparse import Namespace
from collections import defaultdict
from pathlib import Path
from typing import Dict
from typing import List

import numpy as np
import open3d
import pandas as pd
from mmdet3d.structures import Box3DMode
from object_detection_3d.utils import calibration_from_label_id
from object_detection_3d.utils import get_label_path
from object_detection_3d.utils import get_path_component
from object_detection_3d.utils import get_velodyne_path
from object_detection_3d.utils import get_velodyne_pcd_path
from object_detection_3d.workflow import AnnotatedBoundingBox
from object_detection_3d.workflow import AnnotatedBoundingBox3D
from object_detection_3d.workflow import GroundTruth
from object_detection_3d.workflow import TestSample
from tqdm import tqdm

from kolena.workflow.asset import ImageAsset
from kolena.workflow.asset import PointCloudAsset

# KITTI only supports evaluation of the first three classes but "Van" and "Person_sitting" GTs
# are used to avoid penalizing inferences labeled as "Car" and "Pedestrian" respectively.
SUPPORTED_LABELS = ["Car", "Pedestrian", "Cyclist", "Van", "Person_sitting", "DontCare"]
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


def get_difficulty(truncated: float, occluded: int, height: float):
    # Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
    # Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
    # Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %

    if height >= 25 and occluded <= 2 and truncated <= 0.5:
        return "hard"
    if height >= 25 and occluded <= 1 and truncated <= 0.30:
        return "moderate"
    if height >= 40 and occluded == 0 and truncated <= 0.15:
        return "easy"

    return "unknown"


def is_easy(truncated: float, occluded: int, height: float) -> bool:
    return not (height <= 40 or occluded > 0 or truncated > 0.15)


def is_moderate(truncated: float, occluded: int, height: float) -> bool:
    return not (height <= 25 or occluded > 1 or truncated > 0.30)


def is_hard(truncated: float, occluded: int, height: float) -> bool:
    return not (height <= 25 or occluded > 2 or truncated > 0.5)


def calibrate_velo_to_cam(label_filepath: str, calibration: Dict[str, np.ndarray]) -> pd.DataFrame:
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


def gt_from_label_id(datadir: Path, label_id: str, calibration: Dict[str, np.ndarray]) -> GroundTruth:
    label_filepath = get_label_path(datadir) / f"{label_id}.txt"
    df = calibrate_velo_to_cam(str(label_filepath), calibration)
    bboxes_2d: List[AnnotatedBoundingBox] = []
    bboxes_3d: List[AnnotatedBoundingBox3D] = []
    counts_by_label: Dict[str, int] = defaultdict(int)
    counts_by_difficulty: Dict[str, int] = defaultdict(int)
    for row in df.itertuples():
        height = math.fabs(row.bbox_y1 - row.bbox_y0)
        easy = is_easy(row.truncated, row.occluded, height)
        moderate = is_moderate(row.truncated, row.occluded, height)
        hard = is_hard(row.truncated, row.occluded, height)
        bbox_2d = AnnotatedBoundingBox(
            label=row.type,
            top_left=(row.bbox_x0, row.bbox_y0),
            bottom_right=(row.bbox_x1, row.bbox_y1),
            is_easy=easy,
            is_moderate=moderate,
            is_hard=hard,
        )
        bbox_3d = AnnotatedBoundingBox3D(
            label=row.type,
            dimensions=(row.dim_x, row.dim_y, row.dim_z),
            center=(row.loc_x, row.loc_y, row.loc_z + (row.dim_z / 2.0)),  # translate in z (up) axis to be in center
            rotations=(0.0, 0.0, row.rotation_y),  # Z yaw axis for lidar coordinates
            truncated=row.truncated,
            occluded=row.occluded,
            alpha=row.alpha,
            is_easy=easy,
            is_moderate=moderate,
            is_hard=hard,
        )
        counts_by_label[row.type] += 1
        counts_by_difficulty["easy"] += int(easy)
        counts_by_difficulty["moderate"] += int(moderate)
        counts_by_difficulty["hard"] += int(hard)
        bboxes_2d.append(bbox_2d)
        bboxes_3d.append(bbox_3d)

    return GroundTruth(
        total_objects=len(df),
        n_car=counts_by_label["Car"],
        n_pedestrian=counts_by_label["Pedestrian"],
        n_cyclist=counts_by_label["Cyclist"],
        n_easy=counts_by_difficulty["easy"],
        n_moderate=counts_by_difficulty["moderate"],
        n_hard=counts_by_difficulty["hard"],
        n_unknown=counts_by_difficulty["unknown"],
        bboxes_2d=bboxes_2d,
        bboxes_3d=bboxes_3d,
    )


def test_sample_from_label_id(remote_prefix: str, label_id: str, calib: Dict[str, np.ndarray]) -> TestSample:
    locator = posixpath.join(remote_prefix, get_path_component("image_2").as_posix(), f"{label_id}.png")
    right_path = posixpath.join(remote_prefix, get_path_component("image_3").as_posix(), f"{label_id}.png")
    velodyne_path = posixpath.join(remote_prefix, get_path_component("velodyne_pcd").as_posix(), f"{label_id}.pcd")
    right = ImageAsset(locator=right_path)
    velodyne = PointCloudAsset(locator=velodyne_path)

    return TestSample(
        locator,
        right=right,
        velodyne=velodyne,
        velodyne_to_camera_transformation=calib["velodyne_to_camera"].flatten().tolist(),
        camera_rectification=calib["camera_rectification"].flatten().tolist(),
        image_projection=calib["image_projection"].flatten().tolist(),
    )


def convert_kitti_to_pcd(lidar_file: str, target_file: str) -> None:
    bin_pcd = np.fromfile(lidar_file, dtype=np.float32)
    points = bin_pcd.reshape((-1, 4))[:, 0:3]
    o3d_pcd = open3d.geometry.PointCloud(open3d.utility.Vector3dVector(points))
    open3d.io.write_point_cloud(target_file, o3d_pcd, compressed=True)


def prepare_pcd(datadir: Path) -> None:
    """Convert KITTI velodyne bin format to PCD for visualization"""

    velodyne_path = get_velodyne_path(datadir)
    pcd_path = get_velodyne_pcd_path(datadir)
    if not os.path.exists(pcd_path):
        os.mkdir(pcd_path)

    for bin in glob.glob(f"{velodyne_path}/*.bin"):
        filename = os.path.basename(bin)
        pcd = str(pcd_path / f"{os.path.splitext(filename)[0]}.pcd")
        if not os.path.exists(pcd):
            convert_kitti_to_pcd(bin, pcd)


def prepare_sample_data(args: Namespace) -> None:
    label_files = os.listdir(get_label_path(args.datadir))
    label_ids = [Path(filepath).stem for filepath in label_files]
    test_samples_and_ground_truths = []
    for label_id in tqdm(label_ids, desc="Preparing test samples"):
        calib = calibration_from_label_id(args.datadir, label_id)
        test_samples_and_ground_truths.append(
            (
                dataclasses.asdict(test_sample_from_label_id(args.remote_prefix, label_id, calib)),
                dataclasses.asdict(gt_from_label_id(args.datadir, label_id, calib)),
            ),
        )

    with open(str(args.output), "w") as f:
        json.dump(dict(data=test_samples_and_ground_truths), f)


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("datadir", help="KITTI dataset dir", type=Path)
    parser.add_argument("remote_prefix", help="Prefix of cloud storage of KITTI raw data")
    parser.add_argument("output", help="output file", type=Path)
    args = parser.parse_args()
    return args


def main(args: Namespace) -> int:
    prepare_pcd(args.datadir)
    prepare_sample_data(args)

    return 0


if __name__ == "__main__":
    args = parse_args()
    main(args)
