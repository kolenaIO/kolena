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
from typing import Any
from typing import Dict
from typing import List

import pandas as pd
from instance_segmentation.constants import BUCKET
from instance_segmentation.constants import DATASET
from instance_segmentation.constants import ID_FIELDS
from instance_segmentation.constants import LABEL_TO_COLOR
from instance_segmentation.constants import TASK

from kolena.annotation import LabeledPolygon
from kolena.dataset import upload_dataset


def load_data(df_metadata_csv: pd.DataFrame) -> pd.DataFrame:
    image_to_polygons: Dict[str, List[LabeledPolygon]] = defaultdict(list)
    image_to_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for record in df_metadata_csv.itertuples():
        points = [
            (float(record.min_x), float(record.min_y)),
            (float(record.min_x), float(record.max_y)),
            (float(record.max_x), float(record.max_y)),
            (float(record.max_x), float(record.min_y)),
        ]
        polygon = LabeledPolygon(
            points,
            record.label,
            supercategory=record.supercategory,  # type: ignore[call-arg]
            color=LABEL_TO_COLOR[record.label],  # type: ignore[call-arg]
        )
        image_to_polygons[record.locator].append(polygon)
        metadata = {
            "locator": str(record.locator),
            "height": float(record.height),
            "width": float(record.width),
            "date_captured": str(record.date_captured),
            "brightness": float(record.brightness),
        }
        image_to_metadata[record.locator] = metadata

    df_polygons = pd.DataFrame(list(image_to_polygons.items()), columns=["locator", "ground_truths"])
    df_metadata = pd.DataFrame.from_dict(image_to_metadata, orient="index").reset_index(drop=True)
    return df_polygons.merge(df_metadata, on="locator")


def run(args: Namespace) -> None:
    df_metadata_csv = pd.read_csv(
        f"s3://{BUCKET}/{DATASET}/{TASK}/raw/{DATASET}.csv",
        storage_options={"anon": True},
    )
    df_metadata = load_data(df_metadata_csv)
    upload_dataset(args.dataset, df_metadata, id_fields=ID_FIELDS)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("--dataset", type=str, default=DATASET, help="Optionally specify a custom dataset name to upload.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
