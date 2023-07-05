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
import random
import uuid
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import pytest
from object_detection_3d.seed_test_run import seed_test_run
from object_detection_3d.seed_test_suite import seed_test_suite
from object_detection_3d.workflow import AnnotatedBoundingBox3D
from object_detection_3d.workflow import GroundTruth
from object_detection_3d.workflow import TestSample

from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.asset import PointCloudAsset


TEST_SUITE_NAME = "KITTI 3D Object Detection Smoke Test Suite"


def dummy_test_samples() -> List[Tuple[TestSample, GroundTruth]]:
    random_prefix = uuid.uuid4()
    n_samples = 10
    n_gt = 5
    return [
        (
            TestSample(
                locator=f"s3://{random_prefix}/kitti/training/image_2/{i:05d}.png",
                velodyne=PointCloudAsset(locator=f"s3://{random_prefix}/kitti/training/velodyne_pcd/{i:05d}.pcd"),
                velodyne_to_camera_transformation=[random.random() for _ in range(4) for _ in range(4)],
                camera_rectification=[random.random() for _ in range(4) for _ in range(4)],
                image_projection=[random.random() for _ in range(4) for _ in range(4)],
            ),
            GroundTruth(
                total=n_gt,
                easy_image_bboxes=[],
                easy_velodyne_bboxes=[],
                moderate_image_bboxes=[
                    LabeledBoundingBox(
                        label="Car",
                        top_left=(100 + i * 5, 100 + i * 5),
                        bottom_right=(150 + i * 5, 150 + i * 5),
                    )
                    for i in range(n_gt)
                ],
                moderate_velodyne_bboxes=[
                    AnnotatedBoundingBox3D(
                        label="Car",
                        truncated=0,
                        occluded=0,
                        alpha=math.pi / 2,
                        center=(100 + i * 10, 100 + i * 10, 50 + i * 10),
                        dimensions=(50, 50, 50),
                        rotations=(0, 0, 0),
                        difficulty="moderate",
                    )
                    for i in range(n_gt)
                ],
                hard_image_bboxes=[],
                hard_velodyne_bboxes=[],
                other_image_bboxes=[],
                other_velodyne_bboxes=[],
            ),
        )
        for i in range(n_samples)
    ]


def dummy_results() -> List[Dict[str, Any]]:
    return [
        {
            "label_id": f"{i:05d}",
            "bboxes": [
                {
                    "box": [random.uniform(10, 1000) for _ in range(4)],
                    "box3d": [random.uniform(-2, 8) if i < 6 else random.uniform(-math.pi, math.pi) for i in range(7)],
                    "score": random.random(),
                    "pred": random.choice(["Pedestrian", "Car", "Cyclist"]),
                }
                for _ in range(5)
            ],
        }
        for i in range(10)
    ]


def test__seed_test_suite__smoke() -> None:
    seed_test_suite(TEST_SUITE_NAME, dummy_test_samples())


@pytest.mark.depends(on=["test__seed_test_suite__smoke"])
@pytest.skip("expected to fail until evaluation codes can run without cuda")
def test__seed_test_run__smoke() -> None:
    model_name = f"KITTI 3D Object Detection ({uuid.uuid4()})"
    seed_test_run(TEST_SUITE_NAME, model_name, dummy_results())
