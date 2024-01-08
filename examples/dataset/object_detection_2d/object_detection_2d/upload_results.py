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
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import pandas as pd
from commons import BUCKET
from commons import DATASET
from commons import defaultdict
from commons import load_data
from metrics import test_sample_metrics

import kolena
from kolena._experimental.object_detection.utils import filter_inferences
from kolena.dataset import download_dataset
from kolena.dataset import upload_results
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.metrics._geometry import match_inferences_multiclass

MODEL = "coco-2014-val_prediction_complete"
EVAL_CONFIG = {
    "threshold_strategy": 0.5,
    "iou_threshold": 0.5,
    "min_confidence_score": 0.5,
}


def process_inferences(
    pred_df: pd.DataFrame,
    eval_config: Dict[str, Union[float, int]],
) -> pd.DataFrame:
    results: List[Dict[str, Any]] = list()
    for record in pred_df.itertuples():
        ground_truths = [LabeledBoundingBox(box.top_left, box.bottom_right, box.label) for box in record.bounding_boxes]
        inferences = record.raw_inferences
        matches = match_inferences_multiclass(
            ground_truths,
            filter_inferences(inferences, eval_config["min_confidence_score"]),
            mode="pascal",
            iou_threshold=eval_config["iou_threshold"],
        )
        results.append(
            test_sample_metrics(matches, defaultdict(lambda: eval_config["threshold_strategy"])),
        )

    results_df = pd.concat([pd.DataFrame(results), pred_df], axis=1)
    results_df.drop("raw_inferences", axis=1)
    return results_df


def main() -> None:
    kolena.initialize(verbose=True)
    pred_df_csv = pd.read_csv(
        f"s3://{BUCKET}/{DATASET}/results/object_detection/{MODEL}.csv",
        storage_options={"anon": True},
    )
    pred_df = load_data(pred_df_csv, is_pred=True)
    dataset_df = download_dataset(DATASET)
    results_df = process_inferences(pred_df.merge(dataset_df, on="image_id"), EVAL_CONFIG)
    upload_results(DATASET, MODEL, results_df)


if __name__ == "__main__":
    main()
