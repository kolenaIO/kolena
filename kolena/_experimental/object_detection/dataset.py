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
import itertools
from collections import defaultdict
from typing import Any
from typing import cast
from typing import Dict
from typing import List
from typing import Literal
from typing import Union

import pandas as pd

from kolena import dataset
from kolena._experimental.object_detection.utils import compute_optimal_f1_threshold
from kolena._experimental.object_detection.utils import compute_optimal_f1_threshold_multiclass
from kolena._experimental.object_detection.utils import filter_inferences
from kolena.annotation import ScoredLabel
from kolena.errors import IncorrectUsageError
from kolena.metrics import InferenceMatches
from kolena.metrics import match_inferences
from kolena.metrics import match_inferences_multiclass
from kolena.metrics import MulticlassInferenceMatches


def single_class_datapoint_metrics(object_matches: InferenceMatches, thresholds: float) -> Dict[str, Any]:
    tp = [{**gt._to_dict(), **inf._to_dict()} for gt, inf in object_matches.matched if inf.score >= thresholds]
    fp = [inf for inf in object_matches.unmatched_inf if inf.score >= thresholds]
    fn = object_matches.unmatched_gt + [gt for gt, inf in object_matches.matched if inf.score < thresholds]
    scores = [inf["score"] for inf in tp] + [inf.score for inf in fp]
    return dict(
        TP=tp,
        FP=fp,
        FN=fn,
        matched_inference=[{**gt._to_dict(), **inf._to_dict()} for gt, inf in object_matches.matched],
        unmatched_ground_truth=object_matches.unmatched_gt,
        unmatched_inference=object_matches.unmatched_inf,
        count_TP=len(tp),
        count_FP=len(fp),
        count_FN=len(fn),
        has_TP=len(tp) > 0,
        has_FP=len(fp) > 0,
        has_FN=len(fn) > 0,
        max_confidence_above_t=max(scores) if len(scores) > 0 else None,
        min_confidence_above_t=min(scores) if len(scores) > 0 else None,
        thresholds=thresholds,
    )


def multiclass_datapoint_metrics(
    object_matches: MulticlassInferenceMatches,
    thresholds: Dict[str, float],
) -> Dict[str, Any]:
    tp = [
        {**gt._to_dict(), **inf._to_dict()} for gt, inf in object_matches.matched if inf.score >= thresholds[inf.label]
    ]
    fp = [inf for inf in object_matches.unmatched_inf if inf.score >= thresholds[inf.label]]
    fn = [gt for gt, _ in object_matches.unmatched_gt] + [
        gt for gt, inf in object_matches.matched if inf.score < thresholds[inf.label]
    ]
    unmatched_ground_truth = [
        dict(**gt._to_dict(), predicted_label=inf.label, predicted_score=inf.score) if inf else gt
        for gt, inf in object_matches.unmatched_gt
    ]
    confused = [
        dict(**inf._to_dict(), actual_label=gt.label)
        for gt, inf in object_matches.unmatched_gt
        if inf is not None and inf.score >= thresholds[inf.label]
    ]
    scores = [inf["score"] for inf in tp] + [inf.score for inf in fp]
    inference_labels = {inf.label for _, inf in object_matches.matched}.union(
        {inf.label for inf in object_matches.unmatched_inf},
    )
    fields = [
        ScoredLabel(label=label, score=thresholds[label])
        for label in sorted(thresholds.keys())
        if label in inference_labels
    ]
    return dict(
        TP=tp,
        FP=fp,
        FN=fn,
        matched_inference=[{**gt._to_dict(), **inf._to_dict()} for gt, inf in object_matches.matched],
        unmatched_ground_truth=unmatched_ground_truth,
        unmatched_inference=object_matches.unmatched_inf,
        Confused=confused,
        count_TP=len(tp),
        count_FP=len(fp),
        count_FN=len(fn),
        count_Confused=len(confused),
        has_TP=len(tp) > 0,
        has_FP=len(fp) > 0,
        has_FN=len(fn) > 0,
        has_Confused=len(confused) > 0,
        max_confidence_above_t=max(scores) if len(scores) > 0 else None,
        min_confidence_above_t=min(scores) if len(scores) > 0 else None,
        thresholds=fields,
    )


def _compute_metrics(
    pred_df: pd.DataFrame,
    *,
    ground_truth: str,
    inference: str,
    iou_threshold: float = 0.5,
    threshold_strategy: Union[Literal["F1-Optimal"], float, Dict[str, float]] = 0.5,
    min_confidence_score: float = 0.5,
) -> pd.DataFrame:
    """
    Compute metrics for object detection.

    :param df: Dataframe for model results.
    :param ground_truth: Column name for ground truth object annotations
    :param inference: Column name for inference object annotations
    :param iou_threshold: The [IoU ↗](../../metrics/iou.md) threshold, defaulting to `0.5`.
    :param threshold_strategy: The confidence threshold strategy. It can either be a fixed confidence threshold such
        as `0.5` or `0.75`, or the F1-optimal threshold.
    :param min_confidence_score: The minimum confidence score to consider for the evaluation. This is usually set to
        reduce noise by excluding inferences with low confidence score.
    """
    is_multiclass = _check_multiclass(pred_df[ground_truth], pred_df[inference])
    match_fn = match_inferences_multiclass if is_multiclass else match_inferences

    idx = {name: i for i, name in enumerate(list(pred_df), start=1)}

    all_object_matches: Union[List[MulticlassInferenceMatches], List[InferenceMatches]] = []
    for record in pred_df.itertuples():
        ground_truths = record[idx[ground_truth]]
        inferences = record[idx[inference]]
        all_object_matches.append(
            match_fn(  # type: ignore[arg-type]
                ground_truths,
                filter_inferences(inferences, min_confidence_score),
                mode="pascal",
                iou_threshold=iou_threshold,
            ),
        )

    thresholds: dict[str, float]

    if is_multiclass:
        if isinstance(threshold_strategy, dict):
            thresholds = threshold_strategy
        elif isinstance(threshold_strategy, float):
            thresholds = defaultdict(lambda: threshold_strategy)
        else:
            thresholds = compute_optimal_f1_threshold_multiclass(
                cast(List[MulticlassInferenceMatches], all_object_matches),
            )
        results = [
            multiclass_datapoint_metrics(cast(MulticlassInferenceMatches, matches), thresholds)
            for matches in all_object_matches
        ]
    else:
        if isinstance(threshold_strategy, dict) and threshold_strategy:
            threshold = next(iter(threshold_strategy.values()))
        elif isinstance(threshold_strategy, float):
            threshold = threshold_strategy
        else:
            threshold = compute_optimal_f1_threshold(cast(List[InferenceMatches], all_object_matches))
        results = [
            single_class_datapoint_metrics(cast(InferenceMatches, matches), threshold) for matches in all_object_matches
        ]

    return pd.concat([pd.DataFrame(results), pred_df], axis=1)


def _check_multiclass(ground_truth: pd.Series, inference: pd.Series) -> bool:
    try:
        labels = {x.label for x in itertools.chain.from_iterable(ground_truth)}.union(
            {x.label for x in itertools.chain.from_iterable(inference)},
        )
        return len(labels) >= 2
    except AttributeError:
        return False


def _validate_column_present(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise IncorrectUsageError(f"Missing column '{col}'")


def upload_object_detection_results(
    dataset_name: str,
    model_name: str,
    df: pd.DataFrame,
    *,
    ground_truths_field: str = "ground_truths",
    raw_inferences_field: str = "raw_inferences",
    iou_threshold: float = 0.5,
    threshold_strategy: Union[Literal["F1-Optimal"], float, Dict[str, float]] = "F1-Optimal",
    min_confidence_score: float = 0.01,
) -> None:
    """
    Compute metrics and upload results of the model for the dataset.

    Dataframe `df` should include a `locator` column that would match to that of corresponding datapoint. Column
    :inference in the Dataframe `df` should be a list of scored [`BoundingBoxes`][kolena.annotation.BoundingBox].

    :param dataset_name: Dataset name.
    :param model_name: Model name.
    :param df: Dataframe for model results.
    :param ground_truths_field: Field name in datapoint with ground truth bounding boxes,
    defaulting to `"ground_truths"`.
    :param raw_inferences_field: Column in model result DataFrame with raw inference bounding boxes,
    defaulting to `"raw_inferences"`.
    :param iou_threshold: The [IoU ↗](../../metrics/iou.md) threshold, defaulting to `0.5`.
    :param threshold_strategy: The confidence threshold strategy. It can either be a fixed confidence threshold such
        as `0.5` or `0.75`, or `"F1-Optimal"` to find the threshold maximizing F1 score..
    :param min_confidence_score: The minimum confidence score to consider for the evaluation. This is usually set to
        reduce noise by excluding inferences with low confidence score.
    :return:
    """
    eval_config = dict(
        iou_threshold=iou_threshold,
        threshold_strategy=threshold_strategy,
        min_confidence_score=min_confidence_score,
    )
    _validate_column_present(df, raw_inferences_field)

    dataset_df = dataset.download_dataset(dataset_name)
    dataset_df = dataset_df[["locator", ground_truths_field]]
    _validate_column_present(dataset_df, ground_truths_field)

    results_df = _compute_metrics(
        df.merge(dataset_df, on="locator"),
        ground_truth=ground_truths_field,
        inference=raw_inferences_field,
        threshold_strategy=threshold_strategy,
        iou_threshold=iou_threshold,
        min_confidence_score=min_confidence_score,
    )
    results_df.drop(columns=[ground_truths_field], inplace=True)
    dataset.upload_results(dataset_name, model_name, [(eval_config, results_df)])
