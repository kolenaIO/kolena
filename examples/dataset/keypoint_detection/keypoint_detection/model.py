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
import random
from typing import Any
from typing import List
from typing import Tuple

import cv2
import numpy as np
import s3fs

try:
    from retinaface import RetinaFace  # noqa: F401
except ImportError:
    print("Note: Package 'retinaface' not found; install 'retinaface' with `poetry install --extras retina`")


from kolena.annotation import Keypoints
from kolena.annotation import ScoredLabeledBoundingBox


def infer_from_df(record: Any, df: pd.DataFrame) -> Tuple[List[ScoredLabeledBoundingBox], List[Keypoints]]:
    inference = df[df["locator"] == record.locator]
    return inference.iloc[0]["raw_bboxes"], inference.iloc[0]["raw_faces"]


def infer_random(record: Any) -> Tuple[List[ScoredLabeledBoundingBox], List[Keypoints]]:
    def randomize(point: Tuple[float, float]) -> Tuple[float, float]:
        return point[0] + (random.random() - 0.5) * 100, point[1] + (random.random() - 0.5) * 100

    gt_points = record.face.points
    random_bbox = ScoredLabeledBoundingBox(
        top_left=(gt_points[1][0] - random.random() * 100, gt_points[1][1] - random.random() * 100),
        bottom_right=(gt_points[4][0] + random.random() * 100, gt_points[4][1] + random.random() * 100),
        score=random.random(),
        label="random",
    )
    random_face = Keypoints(points=[randomize(pt) for pt in gt_points])
    return [random_bbox], [random_face]
