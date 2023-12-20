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

kolena.initialize(verbose=True)


def infer(locator: str):
    image = download_image(locator)
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
            keypoints = Keypoints(
                points=[
                    pred["landmarks"]["left_eye"],
                    pred["landmarks"]["right_eye"],
                    pred["landmarks"]["nose"],
                    pred["landmarks"]["mouth_left"],
                    pred["landmarks"]["mouth_right"],
                ],
            )
            bboxes.append(bbox)
            faces.append(keypoints)
    except Exception:
        pass  # prediction failure
    return bboxes, faces


df = fetch_dataset("300-W")  # TODO: use const
inferences = []
for record in tqdm(df.itertuples(), total=len(df)):
    pred = infer(record.locator)
    bboxes = [bbox for bbox, _ in pred]
    keypoints = [kp for _, kp in pred]
    inferences.append((record.locator, bboxes, keypoints))

df_pred = pd.DataFrame(inferences, columns=["locator", "bboxes", "keypoints"])


results = []
for record in tqdm(pd.concat((df, df_pred), axis=1).itertuples(), total=len(df)):
    scores = compute_metrics(record)
    results.append(dict(locator=record.locator, **scores))


df_results = pd.DataFrame.from_records(results)
test("300-W", "RetinaFace-results", df_results)
