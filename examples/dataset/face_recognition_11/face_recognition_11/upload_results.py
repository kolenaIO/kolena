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
import json
from argparse import ArgumentParser
from argparse import Namespace

import numpy as np
import pandas as pd
import s3fs
from face_recognition_11.constants import BUCKET
from face_recognition_11.constants import DATASET
from face_recognition_11.constants import WORKFLOW
from face_recognition_11.image import FRImageAsset
from face_recognition_11.metrics import compute_alignment_metrics
from face_recognition_11.metrics import compute_detection_metrics
from face_recognition_11.metrics import compute_pairwise_recognition_merics
from face_recognition_11.metrics import compute_recognition_merics
from face_recognition_11.metrics import compute_recognition_threshold

import kolena
from kolena.annotation import BoundingBox
from kolena.annotation import Keypoints
from kolena.dataset import download_dataset
from kolena.dataset import upload_results


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    df_dataset = download_dataset(args.dataset)
    df_results = pd.read_csv(f"s3://{BUCKET}/{DATASET}/{WORKFLOW}/results/raw/{args.model}_{args.detector}.csv")
    fs = s3fs.S3FileSystem()
    with fs.open(f"{BUCKET}/{DATASET}/{WORKFLOW}/results/raw/{args.model}_{args.detector}.config.json", "rb") as f:
        eval_config = json.load(f)

    similarity_threshold = compute_recognition_threshold(df_results, eval_config["false_match_rate"])

    results = []
    for locator, df_locator_results in df_results.groupby("locator_1"):
        pairs = [
            FRImageAsset(
                locator=record.locator_2,
                is_match=record.is_match,
                similarity=record.similarity,
                **compute_pairwise_recognition_merics(record.is_match, record.similarity, similarity_threshold),
            )
            for record in df_locator_results.itertuples()
        ]

        record = df_locator_results.iloc[0]
        dataset_record = df_dataset[df_dataset["locator"] == locator].iloc[0]
        bbox = BoundingBox(
            top_left=(record.min_x, record.min_y),
            bottom_right=(record.max_x, record.max_y),
        )

        if np.isnan(record.left_eye_x) or np.isnan(record.right_eye_x):
            keypoints = Keypoints(points=[])
        else:
            keypoints = Keypoints(
                points=[
                    (record.left_eye_x, record.left_eye_y),
                    (record.right_eye_x, record.right_eye_y),
                ],
            )

        results.append(
            dict(
                locator=locator,
                pairs=pairs,
                bbox=bbox,
                keypoints=keypoints,
                **compute_detection_metrics(dataset_record.bbox, bbox, eval_config),
                **compute_alignment_metrics(
                    dataset_record.normalization_factor,
                    dataset_record.keypoints,
                    keypoints,
                    eval_config,
                ),
                **compute_recognition_merics(pairs, similarity_threshold),
            ),
        )

    df_results = pd.DataFrame(results)
    upload_results(args.dataset, f"{args.model}_{args.detector}", [(eval_config, df_results)])


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--model", type=str, choices=["vgg-face", "facenet512"], help="Name of FR model to test.")
    ap.add_argument("--detector", type=str, choices=["mtcnn", "dlib"], help="Name of detector backend to test.")
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Optionally specify a custom dataset name to test.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
