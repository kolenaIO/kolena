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
from collections import defaultdict

import pandas as pd
import s3fs as s3fs
from person_detection.constants import BUCKET
from person_detection.constants import DATASET_DIR
from person_detection.constants import DATASET_NAME
from person_detection.constants import MODELS
from person_detection.constants import TASK_DIR

from kolena._experimental.object_detection import upload_object_detection_results
from kolena.annotation import ScoredLabeledBoundingBox


def to_locator(filename: str) -> str:
    return f"s3://{BUCKET}/{DATASET_DIR}/data/{filename}"


def map_results(images: list[dict], annotations: list[dict], categories: list[dict]) -> pd.DataFrame:
    labels = {category["id"]: category["name"] for category in categories}
    image_locators = {image["id"]: to_locator(image["file_name"]) for image in images}
    results = defaultdict(list)
    for bbox in annotations:
        locator = image_locators[bbox["image_id"]]
        label = labels[bbox["category_id"]]
        x, y, width, height = bbox["bbox"]
        results[locator].append(
            ScoredLabeledBoundingBox(
                top_left=(x, y),
                bottom_right=(x + width, y + height),
                score=bbox["score"],
                label=label,
            ),
        )

    return pd.DataFrame(results.items(), columns=["locator", "raw_inferences"])


def run(args: Namespace) -> None:
    s3 = s3fs.S3FileSystem()
    with s3.open(f"s3://{BUCKET}/{DATASET_DIR}/{TASK_DIR}/results/raw/{args.model}.json") as f:
        coco_results = json.loads(f.read())

    df_pred = map_results(coco_results["images"], coco_results["annotations"], coco_results["categories"])
    sample_count = args.sample_count
    if sample_count:
        df_pred.sort_values(by="locator", inplace=True, ignore_index=True)
        df_pred = df_pred[:sample_count]

    upload_object_detection_results(args.dataset, args.model, df_pred)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=MODELS, help="Name of the model to test.")
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET_NAME,
        help="Optionally specify a custom dataset name to test.",
    )
    run(ap.parse_args())


if __name__ == "__main__":
    main()
