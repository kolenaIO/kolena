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
from keypoint_detection.constants import BUCKET
from keypoint_detection.constants import DATASET
from keypoint_detection.metrics import compute_metrics
from keypoint_detection.model import infer_from_df
from keypoint_detection.model import infer_random
from tqdm import tqdm

from kolena.annotation import Keypoints
from kolena.annotation import ScoredLabeledBoundingBox
from kolena.dataset import download_dataset
from kolena.dataset import upload_results
from kolena.io import dataframe_from_csv


MODELS = ["retinaface", "random"]


def run(args: Namespace) -> None:
    infer = infer_random
    if args.model == "retinaface":
        retinaface_raw_inferences_df = dataframe_from_csv(
            f"s3://{BUCKET}/{DATASET}/results/raw/{args.model}.csv",
            storage_options={"anon": True},
        )

        def infer_retinaface(record: Any) -> Tuple[List[ScoredLabeledBoundingBox], List[Keypoints]]:
            return infer_from_df(record, retinaface_raw_inferences_df)

        infer = infer_retinaface

    df = download_dataset(args.dataset)
    results = []
    for record in tqdm(df.itertuples(), total=len(df)):
        bboxes, faces = infer(record)
        metrics = compute_metrics(record.face, faces, record.normalization_factor)
        results.append(dict(locator=record.locator, raw_bboxes=bboxes, raw_faces=faces, **metrics))

    df_results = pd.DataFrame.from_records(results)
    upload_results(args.dataset, args.model, df_results)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument(
        "model",
        type=str,
        choices=MODELS,
        help="Name of the model to test.",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Optionally specify a custom dataset name to test.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
