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
from argparse import ArgumentParser
from argparse import Namespace
from typing import Any
from typing import List
from typing import Tuple

import pandas as pd
from keypoint_detection.metrics import compute_metrics
from keypoint_detection.model import infer_from_df
from keypoint_detection.model import infer_random
from tqdm import tqdm

import kolena
from kolena.dataset import fetch_dataset
from kolena.dataset import test
from kolena.io import dataframe_from_csv
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints

RETINAFACE_S3_PATH = "s3://kolena-public-examples/300-W/results/raw/retinaface.csv"


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    retinaface_raw_inferences_df = dataframe_from_csv(RETINAFACE_S3_PATH, storage_options={"anon": True})

    def infer_retinaface(rec: Any) -> Tuple[List[BoundingBox], List[Keypoints]]:
        return infer_from_df(rec, retinaface_raw_inferences_df)

    infer = infer_retinaface if args.model == "RetinaFace" else infer_random

    df = fetch_dataset(args.dataset)

    results = []
    for record in tqdm(df.itertuples(), total=len(df)):
        bboxes, faces = infer(record)
        metrics = compute_metrics(record.face, faces, record.normalization_factor)
        results.append(dict(locator=record.locator, raw_bboxes=bboxes, raw_faces=faces, **metrics))

    df_results = pd.DataFrame.from_records(results)
    test(args.dataset, args.model, df_results)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=["RetinaFace", "random"], help="Name of model to test.")
    ap.add_argument("dataset", nargs="?", default="300-W", help="Name of dataset to use for testing.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
