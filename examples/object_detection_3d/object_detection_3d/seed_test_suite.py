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
import math
import os
import posixpath
from argparse import ArgumentParser
from pathlib import Path
from typing import List
from typing import Tuple

import pandas as pd
from object_detection_3d.utils import calibration_from_label_id
from object_detection_3d.utils import get_label_path
from object_detection_3d.utils import get_lidar_label_path
from object_detection_3d.utils import get_path_component
from object_detection_3d.utils import LABEL_FILE_COLUMNS
from object_detection_3d.workflow import AnnotatedBoundingBox
from object_detection_3d.workflow import AnnotatedBoundingBox3D
from object_detection_3d.workflow import GroundTruth
from object_detection_3d.workflow import TestCase
from object_detection_3d.workflow import TestSample
from object_detection_3d.workflow import TestSuite
from tqdm import tqdm

import kolena
from kolena.workflow.asset import ImageAsset
from kolena.workflow.asset import PointCloudAsset


# KITTI only supports evaluation of the first three classes but "Van" and "Person_sitting" GTs
# are used to avoid penalizing inferences labeled as "Car" and "Pedestrian" respectively.
SUPPORTED_LABELS = ["Car", "Pedestrian", "Cyclist", "Van", "Person_sitting", "DontCare"]
DEFAULT_TEST_SUITE_NAME = "KITTI 3D Object Detection :: training :: metrics"


def get_difficulty(truncated: float, occluded: int, height: float):
    # Easy: Min. bounding box height: 40 Px, Max. occlusion level: Fully visible, Max. truncation: 15 %
    # Moderate: Min. bounding box height: 25 Px, Max. occlusion level: Partly occluded, Max. truncation: 30 %
    # Hard: Min. bounding box height: 25 Px, Max. occlusion level: Difficult to see, Max. truncation: 50 %

    if height >= 40 and occluded == 0 and truncated <= 0.15:
        return "easy"
    if height >= 25 and occluded <= 1 and truncated <= 0.30:
        return "moderate"
    if height >= 25 and occluded <= 2 and truncated <= 0.5:
        return "hard"

    return "unknown"


def gt_from_label_id(datadir: Path, label_id: str) -> GroundTruth:
    label_filepath = get_lidar_label_path(datadir) / f"{label_id}.txt"
    locator = str(label_filepath)
    df = pd.read_csv(locator, delimiter=" ", header=None, names=LABEL_FILE_COLUMNS)
    df = df[df["type"].isin(SUPPORTED_LABELS)]
    bboxes_2d: List[AnnotatedBoundingBox] = []
    bboxes_3d: List[AnnotatedBoundingBox3D] = []
    for row in df.itertuples():
        difficulty = get_difficulty(row.truncated, row.occluded, math.fabs(row.bbox_y1 - row.bbox_y0))
        bbox_2d = AnnotatedBoundingBox(
            label=row.type,
            top_left=(row.bbox_x0, row.bbox_y0),
            bottom_right=(row.bbox_x1, row.bbox_y1),
            difficulty=difficulty,
        )
        bbox_3d = AnnotatedBoundingBox3D(
            label=row.type,
            dimensions=(row.dim_x, row.dim_y, row.dim_z),
            center=(row.loc_x, row.loc_y, row.loc_z + (row.dim_z / 2.0)),  # translate in z (up) axis to be in center
            rotations=(0.0, 0.0, row.rotation_y),  # Z yaw axis for lidar coordinates
            truncated=row.truncated,
            occluded=row.occluded,
            alpha=row.alpha,
            difficulty=difficulty,
        )
        bboxes_2d.append(bbox_2d)
        bboxes_3d.append(bbox_3d)

    return GroundTruth(
        total_objects=len(df),
        bboxes_2d=bboxes_2d,
        bboxes_3d=bboxes_3d,
    )


def test_sample_from_label_id(datadir: Path, remote_prefix: str, label_id: str) -> TestSample:
    locator = posixpath.join(remote_prefix, get_path_component("image_2").as_posix(), f"{label_id}.png")
    right_path = posixpath.join(remote_prefix, get_path_component("image_3").as_posix(), f"{label_id}.png")
    velodyne_path = posixpath.join(remote_prefix, get_path_component("velodyne_pcd").as_posix(), f"{label_id}.pcd")
    right = ImageAsset(locator=right_path)
    velodyne = PointCloudAsset(locator=velodyne_path)
    calib = calibration_from_label_id(datadir, label_id)

    return TestSample(
        locator,
        right=right,
        velodyne=velodyne,
        velodyne_to_camera_transformation=calib["velodyne_to_camera"].flatten().tolist(),
        camera_rectification=calib["camera_rectification"].flatten().tolist(),
        image_projection=calib["image_projection"].flatten().tolist(),
    )


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--datadir", help="Data dir", type=Path, required=True)
    parser.add_argument("--remote-prefix", help="Prefix of cloud storage of KITTI raw data", required=True)
    parser.add_argument("--test-suite", help="Name of test suite", default=DEFAULT_TEST_SUITE_NAME)
    args = parser.parse_args()
    return args


def seed_test_suite(test_suite_name: str, test_samples: List[Tuple[TestSample, GroundTruth]]):
    kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

    test_cases = TestCase.init_many([(test_suite_name, test_samples)], reset=True)
    test_suite = TestSuite(test_suite_name, test_cases=test_cases, reset=True)
    print(f"created test suite {test_suite.name}")


def main(args):
    label_files = os.listdir(get_label_path(args.datadir))
    label_ids = [Path(filepath).stem for filepath in label_files]
    test_samples_and_ground_truths = [
        (
            test_sample_from_label_id(args.datadir, args.remote_prefix, label_id),
            gt_from_label_id(args.datadir, label_id),
        )
        for label_id in tqdm(label_ids, desc="Preparing test samples")
    ]
    seed_test_suite(args.test_suite, test_samples_and_ground_truths)


if __name__ == "__main__":
    args = parse_args()
    main(args)
