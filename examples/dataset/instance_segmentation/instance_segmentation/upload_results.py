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
from instance_segmentation.constants import BUCKET
from instance_segmentation.constants import DATASET
from instance_segmentation.constants import MODELS
from instance_segmentation.constants import TASK

from kolena._experimental.instance_segmentation import upload_instance_segmentation_results
from kolena.annotation import ScoredLabeledPolygon


def load_data(df_pred_csv: pd.DataFrame) -> pd.DataFrame:
    image_to_polygons: dict[str, list[ScoredLabeledPolygon]] = defaultdict(list)

    for record in df_pred_csv.itertuples():
        points = [
            (float(record.min_x), float(record.min_y)),
            (float(record.min_x), float(record.max_y)),
            (float(record.max_x), float(record.max_y)),
            (float(record.max_x), float(record.min_y)),
        ]
        polygon = ScoredLabeledPolygon(points, record.label, record.confidence_score)
        image_to_polygons[record.locator].append(polygon)

    return pd.DataFrame(list(image_to_polygons.items()), columns=["locator", "raw_inferences"])


def run(args: Namespace) -> None:
    pred_df_csv = pd.read_csv(
        f"s3://{BUCKET}/{DATASET}/{TASK}/results/raw/{args.model}.csv",
        storage_options={"anon": True},
    )
    pred_df = load_data(pred_df_csv)

    upload_instance_segmentation_results(args.dataset, args.model, pred_df)


def main() -> None:
    ap = ArgumentParser()
    ap.add_argument("model", type=str, choices=MODELS, help="Name of the model to test.")
    ap.add_argument("--dataset", type=str, default=DATASET, help="Optionally specify a custom dataset name to test.")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
