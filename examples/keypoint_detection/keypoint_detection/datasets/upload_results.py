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
import random
from argparse import ArgumentParser
from argparse import Namespace
from typing import Any
from typing import List
from typing import Tuple

import pandas as pd
from keypoint_detection.datasets.metrics import compute_metrics
from keypoint_detection.datasets.utils import download_image
from retinaface import RetinaFace
from tqdm import tqdm

import kolena
from kolena._experimental.dataset import fetch_dataset
from kolena._experimental.dataset import test
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints


def infer_retinaface(record: Any) -> Tuple[List[BoundingBox], List[Keypoints]]:
    image = download_image(record.locator)
    predictions = RetinaFace.detect_faces(image)
    bboxes, faces = [], []
    try:
        for face_label, pred in predictions.items():
            bbox = BoundingBox(
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


def infer_random(record: Any) -> Tuple[List[BoundingBox], List[Keypoints]]:
    def randomize(point: Tuple[float, float]) -> Tuple[float, float]:
        return point[0] + (random.random() - 0.5) * 100, point[1] + (random.random() - 0.5) * 100

    gt_points = record.face.points
    random_bbox = BoundingBox(
        top_left=(gt_points[1][0] - random.random() * 100, gt_points[1][1] - random.random() * 100),
        bottom_right=(gt_points[4][0] + random.random() * 100, gt_points[4][1] + random.random() * 100),
        score=random.random(),
        label="random",
    )
    random_face = Keypoints(points=[randomize(pt) for pt in gt_points])
    return [random_bbox], [random_face]


def run(args: Namespace) -> int:
    kolena.initialize(verbose=True)
    infer = infer_retinaface if args.model == "RetinaFace" else infer_random
    df = fetch_dataset(args.dataset)

    inferences = []
    for record in tqdm(df.itertuples(), total=len(df)):
        bboxes, faces = infer(record)
        inferences.append((bboxes, faces))

    results = []
    for record, (bboxes, faces) in tqdm(zip(df.itertuples(), inferences), total=len(df)):
        metrics = compute_metrics(record.face, faces, record.normalization_factor)
        results.append(dict(locator=record.locator, raw_bboxes=bboxes, raw_faces=faces, **metrics))

    df_results = pd.DataFrame.from_records(results)
    test(args.dataset, args.model, df_results)
    return 0


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=["RetinaFace", "random"], help="Name of model to test.")
    ap.add_argument("dataset", nargs="?", default="300-W", help="Name of dataset to use for testing.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
