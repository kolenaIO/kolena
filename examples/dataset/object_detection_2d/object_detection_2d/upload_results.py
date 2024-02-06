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
from collections import defaultdict

import pandas as pd
from object_detection_2d.constants import BUCKET
from object_detection_2d.constants import DATASET
from object_detection_2d.constants import EVAL_CONFIG
from object_detection_2d.constants import MODELS

import kolena
from kolena._experimental.object_detection import upload_object_detection_results
from kolena.annotation import ScoredLabeledBoundingBox


def load_data(df_pred_csv: pd.DataFrame) -> pd.DataFrame:
    image_to_boxes: dict[str, list[ScoredLabeledBoundingBox]] = defaultdict(list)

    for record in df_pred_csv.itertuples():
        coords = (float(record.min_x), float(record.min_y)), (float(record.max_x), float(record.max_y))
        bounding_box = ScoredLabeledBoundingBox(*coords, record.label, record.confidence_score)
        image_to_boxes[record.locator].append(bounding_box)

    return pd.DataFrame(list(image_to_boxes.items()), columns=["locator", "raw_inferences"])


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)

    pred_df_csv = pd.read_csv(
        f"s3://{BUCKET}/{DATASET}/results/raw/{args.model}.csv",
        storage_options={"anon": True},
    )
    pred_df = load_data(pred_df_csv)

    upload_object_detection_results(
        args.dataset,
        args.model,
        pd.DataFrame(pred_df),
        ground_truth="bounding_boxes",
        inference="raw_inferences",
        iou_threshold=EVAL_CONFIG["iou_threshold"],
        threshold_strategy=EVAL_CONFIG["threshold_strategy"],
        min_confidence_score=EVAL_CONFIG["min_confidence_score"],
    )


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=MODELS, help="Name of the model to test.")
    ap.add_argument("--dataset", type=str, default=DATASET, help="Optionally specify a custom dataset name to test.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
