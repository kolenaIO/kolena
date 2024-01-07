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

from kolena.annotation import Keypoints
from kolena.annotation import ScoredLabeledBoundingBox

try:
    from retinaface import RetinaFace  # noqa: F401
except ImportError:
    print("Note: Package 'retinaface' not found; install 'retinaface' with `poetry install --extras retina`")


def download_image(locator: str) -> np.ndarray:
    s3 = s3fs.S3FileSystem(anon=True)
    with s3.open(locator, "rb") as f:
        image_arr = np.asarray(bytearray(f.read()), dtype="uint8")
        image = cv2.imdecode(image_arr, cv2.IMREAD_COLOR)
        return image


def infer_retinaface(record: Any) -> Tuple[List[ScoredLabeledBoundingBox], List[Keypoints]]:
    image = download_image(record.locator)
    predictions = RetinaFace.detect_faces(image)
    bboxes, faces = [], []
    try:
        for face_label, pred in predictions.items():
            bbox = ScoredLabeledBoundingBox(
                top_left=pred["facial_area"][:2],
                bottom_right=pred["facial_area"][2:],
                score=pred["score"],
                label=face_label,
            )
            face = Keypoints(
                points=[
                    pred["landmarks"]["left_eye"],
                    pred["landmarks"]["right_eye"],
                    pred["landmarks"]["nose"],
                    pred["landmarks"]["mouth_left"],
                    pred["landmarks"]["mouth_right"],
                ],
            )
            bboxes.append(bbox)
            faces.append(face)
    except Exception:
        pass  # prediction failure
    return bboxes, faces


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
