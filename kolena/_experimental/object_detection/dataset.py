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
from collections import defaultdict
from typing import Any
from typing import Dict
from typing import List
from typing import Union

import pandas as pd

from kolena import dataset
from kolena._experimental.object_detection.utils import filter_inferences
from kolena.annotation import LabeledBoundingBox
from kolena.annotation import ScoredLabel
from kolena.workflow.metrics import match_inferences_multiclass
from kolena.workflow.metrics import MulticlassInferenceMatches

EVAL_CONFIG = {
    "threshold_strategy": 0.5,
    "iou_threshold": 0.5,
    "min_confidence_score": 0.5,
}


def format_data(df_source: pd.DataFrame) -> pd.DataFrame:
    image_to_boxes: Dict[str, List[LabeledBoundingBox]] = defaultdict(list)
    image_to_metadata: Dict[str, Dict[str, Any]] = defaultdict(dict)

    for record in df_source.itertuples():
        coords = (float(record.min_x), float(record.min_y)), (float(record.max_x), float(record.max_y))
        bounding_box = LabeledBoundingBox(*coords, record.label)
        image_to_boxes[record.locator].append(bounding_box)
        metadata = {
            "locator": str(record.locator),
            "height": float(record.height),
            "width": float(record.width),
            "date_captured": str(record.date_captured),
            "brightness": float(record.brightness),
        }
        image_to_metadata[record.locator] = metadata

    df_boxes = pd.DataFrame(list(image_to_boxes.items()), columns=["locator", "bounding_boxes"])
    df_metadata = pd.DataFrame.from_dict(image_to_metadata, orient="index").reset_index(drop=True)
    return df_boxes.merge(df_metadata, on="locator")


def upload_dataset(name: str, df: pd.DataFrame, eval_config: dict) -> None:
    """
    One bounding box per row in the format of
    locator,min_x,max_x,min_y,max_y

    :param name:
    :param df:
    :param eval_config:
    :return:
    """
    prepared_df = format_data(df)
    dataset.upload_dataset(name, prepared_df)


def datapoint_metrics(
    bbox_matches: MulticlassInferenceMatches,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    tp = [inf for _, inf in bbox_matches.matched if inf.score >= thresholds[inf.label]]
    fp = [inf for inf in bbox_matches.unmatched_inf if inf.score >= thresholds[inf.label]]
    fn = [gt for gt, _ in bbox_matches.unmatched_gt] + [
        gt for gt, inf in bbox_matches.matched if inf.score < thresholds[inf.label]
    ]
    confused = [inf for _, inf in bbox_matches.unmatched_gt if inf is not None and inf.score >= thresholds[inf.label]]
    non_ignored_inferences = tp + fp
    scores = [inf.score for inf in non_ignored_inferences]
    inference_labels = {inf.label for _, inf in bbox_matches.matched} | {
        inf.label for inf in bbox_matches.unmatched_inf
    }
    fields = [
        ScoredLabel(label=label, score=thresholds[label])
        for label in sorted(thresholds.keys())
        if label in inference_labels
    ]
    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "matched_inference": [inf for _, inf in bbox_matches.matched],
        "unmatched_ground_truth": [gt for gt, _ in bbox_matches.unmatched_gt],
        "unmatched_inference": bbox_matches.unmatched_inf,
        "Confused": confused,
        "count_TP": len(tp),
        "count_FP": len(fp),
        "count_FN": len(fn),
        "count_Confused": len(confused),
        "has_TP": len(tp) > 0,
        "has_FP": len(fp) > 0,
        "has_FN": len(fn) > 0,
        "has_Confused": len(confused) > 0,
        "ignored": False,
        "max_confidence_above_t": max(scores) if len(scores) > 0 else None,
        "min_confidence_above_t": min(scores) if len(scores) > 0 else None,
        "thresholds": fields,
    }


def compute_metrics(
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
            datapoint_metrics(matches, defaultdict(lambda: eval_config["threshold_strategy"])),
        )

    results_df = pd.concat([pd.DataFrame(results), pred_df], axis=1)
    return results_df.drop(["bounding_boxes"], axis=1)


def upload_results(dataset_name: str, model_name: str, pred_df: pd.DataFrame) -> None:
    """
    One bounding box per row
    locator,label,confidence_score,min_x,min_y,max_x,max_y

    :param dataset_name:
    :param model_name:
    :param pred_df:
    :return:
    """
    dataset_df = dataset.download_dataset(dataset_name)[["locator", "bounding_boxes"]]
    results_df = compute_metrics(pred_df.merge(dataset_df, on="locator"), EVAL_CONFIG)
    dataset.upload_results(dataset_name, model_name, results_df)
