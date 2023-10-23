# Copyright 2021-2023 Kolena Inc.
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
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

import numpy as np
import pandas as pd
from object_detection_2d.constants import DATASET
from object_detection_2d.constants import S3_MODEL_INFERENCE_PREFIX
from object_detection_2d.constants import TRANSPORTATION_LABELS
from object_detection_2d.constants import WORKFLOW

import kolena
from kolena._experimental.dataset import test
from kolena._experimental.dataset._evaluation import fetch_evaluation_results
from kolena._experimental.dataset._evaluation import fetch_inferences
from kolena._experimental.dataset._evaluation import INFER_FUNC_TYPE
from kolena._experimental.object_detection import ThresholdConfiguration
from kolena._experimental.object_detection.utils import compute_optimal_f1_threshold_multiclass
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.metrics import match_inferences_multiclass
from kolena.workflow.metrics import MulticlassInferenceMatches

MODEL_LIST: Dict[str, str] = {
    "yolo_r": f"YOLOR-D6 (modified CSP, {WORKFLOW})",
    "yolo_x": f"YOLOX (modified CSP-v5, {WORKFLOW})",
    "mask_cnn": f"Mask R-CNN (Inception-ResNet-v2, {WORKFLOW})",
    "faster_rcnn": f"Faster R-CNN (Inception-ResNet-v2, {WORKFLOW})",
    "yolo_v4s": f"Scaled YOLOv4 (CSP-DarkNet-53, {WORKFLOW})",
    "yolo_v3": f"YOLOv3 (DarkNet-53, {WORKFLOW})",
}


def model_alias_to_data_path(alias: str) -> str:
    return S3_MODEL_INFERENCE_PREFIX + alias + "/coco-2014-val_prediction_attribution_2.0_transportation.csv"


def load_results(model_alias: str) -> pd.DataFrame:
    print("loading csv of inferences from S3...")

    try:
        df_results = pd.read_csv(
            model_alias_to_data_path(model_alias),
            dtype={
                "locator": object,
                "label": object,
                "confidence_score": object,
                "min_x": object,
                "min_y": object,
                "max_x": object,
                "max_y": object,
            },
        )
    except OSError as e:
        print(e, "\nPlease ensure you have set up AWS credentials.")
        exit()

    # filter for transportation inferences
    df_results = df_results[df_results.label.isin(TRANSPORTATION_LABELS)]

    # create metadata by image with bboxes
    df_results["bbox"] = df_results.apply(
        lambda row: ScoredLabeledBoundingBox(
            label=row["label"],
            score=float(row["confidence_score"]),
            top_left=(float(row["min_x"]), float(row["min_y"])),
            bottom_right=(float(row["max_x"]), float(row["max_y"])),
        ),
        axis=1,
    )
    metadata_by_image = (
        df_results[["locator", "bbox"]]
        .groupby("locator")["bbox"]
        .apply(list)
        .reset_index(
            name="bboxes",
        )
    )
    metadata_by_image["ignored"] = False

    return metadata_by_image


# transforms test samples into inferences using a dataframe
def get_stored_inferences(
    metadata_by_image: pd.DataFrame,
) -> INFER_FUNC_TYPE:
    # a function that returns inference dataframe from datapoint dataframe
    def infer(datapoints: pd.DataFrame) -> pd.DataFrame:
        inferences = datapoints[["locator"]].merge(metadata_by_image, on="locator")
        return inferences

    return infer


def datapoint_metrics_ignored() -> Dict:
    return dict(
        TP=[],
        FP=[],
        FN=[],
        Confused=[],
        count_TP=0,
        count_FP=0,
        count_FN=0,
        count_Confused=0,
        has_TP=False,
        has_FP=False,
        has_FN=False,
        has_Confused=False,
        ignored=True,
        max_confidence_above_t=None,
        min_confidence_above_t=None,
        thresholds=[],
    )


def datapoint_metrics(bbox_matches: MulticlassInferenceMatches, thresholds: Dict[str, float]) -> Dict:
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
        ScoredClassificationLabel(label=label, score=thresholds[label])
        for label in sorted(thresholds.keys())
        if label in inference_labels
    ]
    return dict(
        TP=tp,
        FP=fp,
        FN=fn,
        Confused=confused,
        count_TP=len(tp),
        count_FP=len(fp),
        count_FN=len(fn),
        count_Confused=len(confused),
        has_TP=len(tp) > 0,
        has_FP=len(fp) > 0,
        has_FN=len(fn) > 0,
        has_Confused=len(confused) > 0,
        ignored=False,
        max_confidence_above_t=max(scores) if len(scores) > 0 else None,
        min_confidence_above_t=min(scores) if len(scores) > 0 else None,
        thresholds=fields,
    )


def filter_inferences(
    inferences: List[ScoredLabeledBoundingBox],
    confidence_score: Optional[float] = None,
    labels: Optional[Set[str]] = None,
) -> List[ScoredLabeledBoundingBox]:
    filtered_by_confidence = (
        [inf for inf in inferences if inf.score >= confidence_score] if confidence_score else inferences
    )
    if labels is None:
        return filtered_by_confidence
    return [inf for inf in filtered_by_confidence if inf.label in labels]


def get_confidence_thresholds(
    configuration: ThresholdConfiguration,
    data: List[Dict],
) -> Dict:
    if configuration.threshold_strategy == "F1-Optimal":
        return compute_f1_optimal_thresholds(configuration, data)
    else:
        return defaultdict(lambda: configuration.threshold_strategy)


def compute_image_metrics(
    gt_bboxes: List[LabeledBoundingBox],
    gt_ignored_bboxes: List[LabeledBoundingBox],
    inf_bboxes: List[ScoredLabeledBoundingBox],
    inf_ignored: bool,
    configuration: ThresholdConfiguration,
    thresholds: Dict,
) -> Dict:
    if inf_ignored:
        return datapoint_metrics_ignored()

    filtered_inferences = filter_inferences(
        inferences=inf_bboxes,
        confidence_score=configuration.min_confidence_score,
    )
    bbox_matches: MulticlassInferenceMatches = match_inferences_multiclass(
        gt_bboxes,
        filtered_inferences,
        ignored_ground_truths=gt_ignored_bboxes,
        mode="pascal",
        iou_threshold=configuration.iou_threshold,
    )

    return datapoint_metrics(bbox_matches, thresholds)


def compute_f1_optimal_thresholds(
    configuration: ThresholdConfiguration,
    data: List[Dict],
) -> Dict:
    all_bbox_matches = [
        match_inferences_multiclass(
            x["gt_bboxes"],
            filter_inferences(inferences=x["inf_bboxes"], confidence_score=configuration.min_confidence_score),
            ignored_ground_truths=x["gt_ignored_bboxes"],
            mode="pascal",
            iou_threshold=configuration.iou_threshold,
        )
        for x in data
        if not x["inf_ignored"]
    ]

    optimal_thresholds = compute_optimal_f1_threshold_multiclass(all_bbox_matches)
    return defaultdict(
        lambda: configuration.min_confidence_score,
        optimal_thresholds,
    )


def eval_func(datapoints: pd.DataFrame, inferences: pd.DataFrame, eval_config: ThresholdConfiguration) -> pd.DataFrame:
    data = pd.DataFrame()
    data["gt_bboxes"] = datapoints["bboxes"].apply(
        lambda arr: [LabeledBoundingBox(label=x.label, top_left=x.top_left, bottom_right=x.bottom_right) for x in arr],
    )
    data["gt_ignored_bboxes"] = datapoints["ignored_bboxes"]
    data["inf_bboxes"] = inferences["bboxes"].apply(
        lambda arr: [
            ScoredLabeledBoundingBox(score=x.score, label=x.label, top_left=x.top_left, bottom_right=x.bottom_right)
            for x in arr
        ],
    )
    data["inf_ignored"] = inferences["ignored"]
    thresholds = get_confidence_thresholds(eval_config, data.to_dict("records"))
    metrics = data.apply(
        lambda x: compute_image_metrics(
            x["gt_bboxes"],
            x["gt_ignored_bboxes"],
            x["inf_bboxes"],
            x["inf_ignored"],
            eval_config,
            thresholds,
        ),
        axis=1,
        result_type="expand",
    ).replace(
        np.nan,
        None,
    )
    return metrics


def seed_evaluation(
    model_full_name: str,
    dataset_name: str,
    groups_df: pd.DataFrame,
) -> None:
    # runs the evaluation
    # deliberately split into 2 invocations for demo purpose, can be combined into one
    # upload inferences
    test(dataset_name, model_full_name, infer=get_stored_inferences(groups_df))
    loaded_datapoints, loaded_inferences = fetch_inferences(dataset_name, model_full_name)
    print(loaded_inferences.shape)

    # compute and upload datapoint metrics
    test(
        dataset_name,
        model_full_name,
        eval=eval_func,
        eval_configs=[
            ThresholdConfiguration(
                threshold_strategy=0.3,
                iou_threshold=0.3,
                min_confidence_score=0.2,
            ),
            ThresholdConfiguration(
                threshold_strategy="F1-Optimal",
                iou_threshold=0.5,
                min_confidence_score=0.0,
            ),
        ],
    )
    # optionally inspect uploaded evaluation results
    loaded_results = fetch_evaluation_results(dataset_name, model_full_name)
    print([(eval, dp.shape, inf.shape, m.shape) for eval, dp, inf, m in loaded_results])


def main(args: Namespace) -> None:
    dataset = args.dataset
    model_alias = args.model
    model_full_name = MODEL_LIST[model_alias]

    kolena.initialize(verbose=True)

    metadata_by_image = load_results(model_alias)
    seed_evaluation(model_full_name, dataset, metadata_by_image)


if __name__ == "__main__":
    ap = ArgumentParser()
    ap.add_argument("model", choices=MODEL_LIST.keys(), help="The alias of the model to test.")
    ap.add_argument(
        "--dataset",
        type=str,
        default=DATASET,
        help="Optionally specify a dataset.",
    )

    main(ap.parse_args())
