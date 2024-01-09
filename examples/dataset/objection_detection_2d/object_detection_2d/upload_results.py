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
from typing import Optional
from typing import Set
from typing import Union

import pandas as pd
from object_detection_2d.constants import BUCKET
from object_detection_2d.constants import DATASET
from object_detection_2d.constants import EVAL_CONFIG
from object_detection_2d.constants import MODELS
from object_detection_2d.data_loader import load_data
from object_detection_2d.metrics import test_sample_metrics

import kolena
from kolena.annotation import LabeledBoundingBox
from kolena.annotation import ScoredLabeledBoundingBox
from kolena.annotation import ScoredLabeledPolygon
from kolena.dataset import download_dataset
from kolena.dataset import upload_results
from kolena.workflow.metrics._geometry import match_inferences_multiclass


def _filter_inferences(
    inferences: List[Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon]],
    confidence_score: Optional[float] = None,
    labels: Optional[Set[str]] = None,
) -> List[Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon]]:
    filtered_by_confidence = (
        [inf for inf in inferences if inf.score >= confidence_score] if confidence_score else inferences
    )
    if labels is None:
        return filtered_by_confidence
    return [inf for inf in filtered_by_confidence if inf.label in labels]


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
            _filter_inferences(inferences, eval_config["min_confidence_score"]),
            mode="pascal",
            iou_threshold=eval_config["iou_threshold"],
        )
        results.append(
            test_sample_metrics(matches, defaultdict(lambda: eval_config["threshold_strategy"])),
        )

    results_df = pd.concat([pd.DataFrame(results), pred_df], axis=1)
    results_df = results_df.drop(["bounding_boxes"], axis=1)
    return results_df


def run(args: Namespace) -> None:
    kolena.initialize(verbose=True)
    pred_df_csv = pd.read_csv(
        f"s3://{BUCKET}/{args.dataset}/results/object_detection/coco_models/"
        + f"{args.model}/coco-2014-val_prediction_attribution_2.0_transportation.csv",
        storage_options={"anon": True},
    )
    pred_df = load_data(pred_df_csv, is_pred=True)
    dataset_df = download_dataset(args.dataset)
    results_df = process_inferences(pred_df.merge(dataset_df, on="locator"), EVAL_CONFIG)
    upload_results(args.dataset, args.model, results_df)


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
