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
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Literal
from typing import Optional
from typing import Union

import numpy as np
import pandas as pd
import tqdm

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


def _bbox_matches_and_count_for_one_label(
    match: MulticlassInferenceMatches,
    label: str,
) -> MulticlassInferenceMatches:
    match_matched = []
    match_unmatched_gt = []
    match_unmatched_inf = []
    for gt, inf in match.matched:
        if gt.label == label:
            match_matched.append((gt, inf))
    for gt, inf in match.unmatched_gt:
        if gt.label == label:
            match_unmatched_gt.append((gt, inf))
    for inf in match.unmatched_inf:
        if inf.label == label:
            match_unmatched_inf.append(inf)

    bbox_matches_for_one_label = MulticlassInferenceMatches(
        matched=match_matched,
        unmatched_gt=match_unmatched_gt,
        unmatched_inf=match_unmatched_inf,
    )

    return bbox_matches_for_one_label


def _compute_thresholded_metrics(
    matches: Union[MulticlassInferenceMatches, InferenceMatches],
    thresholds: List[float],
    label: Union[str, None] = None,
) -> List[Dict[str, Any]]:
    metrics = []
    for threshold in thresholds:
        count_tp = sum(1 for gt, inf in matches.matched if inf.score >= threshold)
        count_fp = sum(1 for inf in matches.unmatched_inf if inf.score >= threshold)
        count_fn = len(matches.unmatched_gt) + sum(1 for _, inf in matches.matched if inf.score < threshold)
        if label:
            metrics.append(dict(threshold=threshold, label=label, tp=count_tp, fp=count_fp, fn=count_fn))
        else:
            metrics.append(dict(threshold=threshold, tp=count_tp, fp=count_fp, fn=count_fn))

    return metrics


def _prepare_thresholded_metrics(
    object_matches: MulticlassInferenceMatches,
    thresholds: List[float],
    labels: List[str],
) -> List[Dict[str, Any]]:
    thresholded_metrics = []
    for label in labels:
        class_matches = _bbox_matches_and_count_for_one_label(object_matches, label)
        thresholded_metrics.extend(_compute_thresholded_metrics(class_matches, thresholds, label))

    return thresholded_metrics


def single_class_datapoint_metrics(
    object_matches: InferenceMatches,
    thresholds: float,
    all_thresholds: List[float],
) -> Dict[str, Any]:
    tp = [{**gt._to_dict(), **inf._to_dict()} for gt, inf in object_matches.matched if inf.score >= thresholds]
    fp = [inf for inf in object_matches.unmatched_inf if inf.score >= thresholds]
    fn = object_matches.unmatched_gt + [gt for gt, inf in object_matches.matched if inf.score < thresholds]
    scores = [inf["score"] for inf in tp] + [inf.score for inf in fp]
    labels = _get_labels_from_objects(
        [inf for _, inf in object_matches.matched]
        + [inf for inf in object_matches.unmatched_inf]
        + [gt for gt in object_matches.unmatched_gt],
    )
    label = None if len(labels) == 0 else labels[0]

    thresholded = _compute_thresholded_metrics(object_matches, all_thresholds, label)
    return dict(
        TP=tp,
        FP=fp,
        FN=fn,
        thresholded=thresholded,
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
    all_thresholds: List[float],
) -> Dict[str, Any]:
    tp = [
        {**gt._to_dict(), **inf._to_dict()} for gt, inf in object_matches.matched if inf.score >= thresholds[inf.label]
    ]
    fp = [inf for inf in object_matches.unmatched_inf if inf.score >= thresholds[inf.label]]
    fn = [gt for gt, _ in object_matches.unmatched_gt] + [
        gt for gt, inf in object_matches.matched if inf.score < thresholds[inf.label]
    ]
    confused = [
        dict(**inf._to_dict(), actual_label=gt.label)
        for gt, inf in object_matches.unmatched_gt
        if inf is not None and inf.score >= thresholds[inf.label]
    ]
    scores = [inf["score"] for inf in tp] + [inf.score for inf in fp]
    labels = sorted(
        _get_labels_from_objects(
            [inf for _, inf in object_matches.matched]
            + [inf for inf in object_matches.unmatched_inf]
            + [gt for gt, _ in object_matches.unmatched_gt],
        ),
    )
    inference_labels = _get_labels_from_objects(
        [inf for _, inf in object_matches.matched] + [inf for inf in object_matches.unmatched_inf],
    )
    fields = [
        ScoredLabel(label=label, score=thresholds[label])
        for label in sorted(thresholds.keys())
        if label in inference_labels
    ]
    thresholded = _prepare_thresholded_metrics(object_matches, thresholds=all_thresholds, labels=labels)
    return dict(
        TP=tp,
        FP=fp,
        FN=fn,
        thresholded=thresholded,
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


def _iter_single_class_metrics(
    pred_df: pd.DataFrame,
    all_object_matches: List[InferenceMatches],
    *,
    threshold: float,
    all_thresholds: List[float],
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    for i in tqdm.tqdm(range(0, pred_df.shape[0], batch_size)):
        metrics = [
            single_class_datapoint_metrics(matches, threshold, all_thresholds)
            for matches in all_object_matches[i : i + batch_size]
        ]
        yield pd.concat([pd.DataFrame(metrics), pred_df.reset_index(drop=True)], axis=1)


def _iter_multi_class_metrics(
    pred_df: pd.DataFrame,
    all_object_matches: List[MulticlassInferenceMatches],
    *,
    thresholds: Dict[str, float],
    all_thresholds: List[float],
    batch_size: int,
) -> Iterator[pd.DataFrame]:
    for i in tqdm.tqdm(range(0, len(all_object_matches), batch_size)):
        metrics = [
            multiclass_datapoint_metrics(matches, thresholds, all_thresholds)
            for matches in all_object_matches[i : i + batch_size]
        ]
        yield pd.concat([pd.DataFrame(metrics), pred_df[i : i + batch_size].reset_index(drop=True)], axis=1)


def _compute_metrics(
    pred_df: pd.DataFrame,
    *,
    ground_truth: str,
    inference: str,
    gt_ignore_property: Optional[str] = None,
    iou_threshold: float = 0.5,
    threshold_strategy: Union[Literal["F1-Optimal"], float, Dict[str, float]] = 0.5,
    min_confidence_score: float = 0.5,
    batch_size: int = 10_000,
    required_match_fields: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Compute metrics for object detection.

    :param df: Dataframe for model results.
    :param ground_truth: Column name for ground truth object annotations
    :param inference: Column name for inference object annotations
    :param gt_ignore_property: Field on the ground truth bounding boxes used to determine if the bounding box should be
    ignored. Bounding boxes will be ignored if this field exists and is equal to `True`.
    :param iou_threshold: The [IoU ↗](../../metrics/iou.md) threshold, defaulting to `0.5`.
    :param threshold_strategy: The confidence threshold strategy. It can either be a fixed confidence threshold such
        as `0.5` or `0.75`, or the F1-optimal threshold.
    :param min_confidence_score: The minimum confidence score to consider for the evaluation. This is usually set to
        reduce noise by excluding inferences with low confidence score.
    :param batch_size: number of results to process per iteration.
    :param Optional[List[str]] required_match_fields: Optionally specify a list of fields that must match between
        the inference and ground truth for them to be considered a match.
    """
    is_multiclass = _check_multiclass(pred_df[ground_truth], pred_df[inference])
    match_fn = match_inferences_multiclass if is_multiclass else match_inferences

    idx = {name: i for i, name in enumerate(list(pred_df), start=1)}

    all_object_matches: Union[List[MulticlassInferenceMatches], List[InferenceMatches]] = []
    all_thresholds: List[float] = []
    for record in pred_df.itertuples():
        ground_truths = record[idx[ground_truth]]
        inferences = record[idx[inference]]
        ignored_ground_truths = [
            gt
            for gt in ground_truths
            if gt_ignore_property is not None
            and hasattr(gt, gt_ignore_property)
            and isinstance(getattr(gt, gt_ignore_property), bool)
            and getattr(gt, gt_ignore_property)
        ]
        unignored_ground_truths = [gt for gt in ground_truths if gt not in ignored_ground_truths]
        all_object_matches.append(
            match_fn(  # type: ignore[arg-type]
                unignored_ground_truths,
                filter_inferences(inferences, min_confidence_score),
                ignored_ground_truths=ignored_ground_truths,
                mode="pascal",
                iou_threshold=iou_threshold,
                required_match_fields=required_match_fields,
            ),
        )
        all_thresholds.extend(inf.score for inf in inferences)

    if len(all_thresholds) >= 501:
        all_thresholds = list(np.linspace(min(all_thresholds), max(all_thresholds), 501))
    else:
        all_thresholds = sorted(all_thresholds)

    thresholds: Dict[str, float]
    pred_df.drop(columns=ground_truth, inplace=True)

    if is_multiclass:
        if isinstance(threshold_strategy, dict):
            thresholds = threshold_strategy
        elif isinstance(threshold_strategy, float):
            thresholds = defaultdict(lambda: threshold_strategy)
        else:
            thresholds = compute_optimal_f1_threshold_multiclass(
                cast(List[MulticlassInferenceMatches], all_object_matches),
            )
        yield from _iter_multi_class_metrics(
            pred_df,
            cast(List[MulticlassInferenceMatches], all_object_matches),
            thresholds=thresholds,
            all_thresholds=all_thresholds,
            batch_size=batch_size,
        )
    else:
        if isinstance(threshold_strategy, dict) and threshold_strategy:
            threshold = next(iter(threshold_strategy.values()))
        elif isinstance(threshold_strategy, float):
            threshold = threshold_strategy
        else:
            threshold = compute_optimal_f1_threshold(cast(List[InferenceMatches], all_object_matches))

        yield from _iter_single_class_metrics(
            pred_df,
            cast(List[InferenceMatches], all_object_matches),
            threshold=threshold,
            all_thresholds=all_thresholds,
            batch_size=batch_size,
        )


def _safe_get_label(obj: object) -> Optional[str]:
    if hasattr(obj, "label"):
        return str(obj.label)

    return None


def _get_labels_from_objects(objs: Iterable[object]) -> List[str]:
    maybe_labels = {_safe_get_label(obj) for obj in objs}
    labels = [label for label in maybe_labels if label is not None]
    return labels


def _check_multiclass(ground_truth: pd.Series, inference: pd.Series) -> bool:
    labels = {_safe_get_label(gt) for gt in itertools.chain.from_iterable(_filter_null(ground_truth))}.union(
        {_safe_get_label(inf) for inf in itertools.chain.from_iterable(_filter_null(inference))},
    )
    if None in labels:
        return False

    return len(labels) >= 2


def _filter_null(series: pd.Series) -> pd.Series:
    return series[series.notnull()]


def _validate_column_present(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise IncorrectUsageError(f"Missing column '{col}'")


def _iter_object_detection_results(
    dataset_name: str,
    df: pd.DataFrame,
    *,
    ground_truths_field: str = "ground_truths",
    raw_inferences_field: str = "raw_inferences",
    gt_ignore_property: Optional[str] = None,
    iou_threshold: float = 0.5,
    threshold_strategy: Union[Literal["F1-Optimal"], float, Dict[str, float]] = "F1-Optimal",
    min_confidence_score: float = 0.01,
    batch_size: int = 10_000,
    required_match_fields: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    _validate_column_present(df, raw_inferences_field)

    dataset_df = dataset.download_dataset(dataset_name)
    dataset_df = dataset_df[["locator", ground_truths_field]]
    _validate_column_present(dataset_df, ground_truths_field)

    merged_df = df.merge(dataset_df, on=["locator"])
    return _compute_metrics(
        merged_df,
        ground_truth=ground_truths_field,
        inference=raw_inferences_field,
        gt_ignore_property=gt_ignore_property,
        iou_threshold=iou_threshold,
        threshold_strategy=threshold_strategy,
        min_confidence_score=min_confidence_score,
        batch_size=batch_size,
        required_match_fields=required_match_fields,
    )


def compute_object_detection_results(
    dataset_name: str,
    df: pd.DataFrame,
    *,
    ground_truths_field: str = "ground_truths",
    raw_inferences_field: str = "raw_inferences",
    gt_ignore_property: Optional[str] = None,
    iou_threshold: float = 0.5,
    threshold_strategy: Union[Literal["F1-Optimal"], float, Dict[str, float]] = "F1-Optimal",
    min_confidence_score: float = 0.01,
    batch_size: int = 10_000,
) -> pd.DataFrame:
    """
    Compute metrics of the model for the dataset.

    Dataframe `df` should include a `locator` column that would match to that of corresponding datapoint and
    an `inference` column that should be a list of scored [`BoundingBoxes`][kolena.annotation.BoundingBox].

    :param dataset_name: Dataset name.
    :param df: Dataframe for model results.
    :param ground_truths_field: Field name in datapoint with ground truth bounding boxes,
    defaulting to `"ground_truths"`.
    :param raw_inferences_field: Column in model result DataFrame with raw inference bounding boxes,
    defaulting to `"raw_inferences"`.
    :param gt_ignore_property: Field on the ground truth bounding boxes used to determine if the bounding box should be
    ignored. Bounding boxes will be ignored if this field exists and is equal to `True`.
    :param iou_threshold: The [IoU ↗](../../metrics/iou.md) threshold, defaulting to `0.5`.
    :param threshold_strategy: The confidence threshold strategy. It can either be a fixed confidence threshold such
        as `0.5` or `0.75`, or `"F1-Optimal"` to find the threshold maximizing F1 score.
    :param min_confidence_score: The minimum confidence score to consider for the evaluation. This is usually set to
        reduce noise by excluding inferences with low confidence score.
    :param batch_size: number of results to process per iteration.
    :return: A `DataFrame` of the computed results
    """
    results_iter = _iter_object_detection_results(
        dataset_name,
        df,
        ground_truths_field=ground_truths_field,
        raw_inferences_field=raw_inferences_field,
        gt_ignore_property=gt_ignore_property,
        iou_threshold=iou_threshold,
        threshold_strategy=threshold_strategy,
        min_confidence_score=min_confidence_score,
        batch_size=batch_size,
    )
    return pd.concat(list(results_iter))


def upload_object_detection_results(
    dataset_name: str,
    model_name: str,
    df: pd.DataFrame,
    *,
    ground_truths_field: str = "ground_truths",
    raw_inferences_field: str = "raw_inferences",
    gt_ignore_property: Optional[str] = None,
    iou_threshold: float = 0.5,
    threshold_strategy: Union[Literal["F1-Optimal"], float, Dict[str, float]] = "F1-Optimal",
    min_confidence_score: float = 0.01,
    batch_size: int = 10_000,
    required_match_fields: Optional[List[str]] = None,
) -> None:
    """
    Compute metrics and upload results of the model computed by
    [`compute_object_detection_results`][kolena._experimental.object_detection.compute_object_detection_results]
    for the dataset.

    Dataframe `df` should include a `locator` column that would match to that of corresponding datapoint and
    an `inference` column that should be a list of scored [`BoundingBoxes`][kolena.annotation.BoundingBox].

    :param dataset_name: Dataset name.
    :param model_name: Model name.
    :param df: Dataframe for model results.
    :param ground_truths_field: Field name in datapoint with ground truth bounding boxes,
    defaulting to `"ground_truths"`.
    :param raw_inferences_field: Column in model result DataFrame with raw inference bounding boxes,
    defaulting to `"raw_inferences"`.
    :param gt_ignore_property: Name of a property on the ground truth bounding boxes used to determine if the bounding
    box should be ignored. Bounding boxes will be ignored if this property exists and is equal to `True`.
    :param iou_threshold: The [IoU ↗](../../metrics/iou.md) threshold, defaulting to `0.5`.
    :param threshold_strategy: The confidence threshold strategy. It can either be a fixed confidence threshold such
        as `0.5` or `0.75`, or `"F1-Optimal"` to find the threshold maximizing F1 score.
    :param min_confidence_score: The minimum confidence score to consider for the evaluation. This is usually set to
        reduce noise by excluding inferences with low confidence score.
    :param batch_size: number of results to process per iteration.
    :param Optional[List[str]] required_match_fields: Optionally specify a list of fields that must match between
        the inference and ground truth for them to be considered a match.
    :return:
    """
    eval_config = dict(
        iou_threshold=iou_threshold,
        threshold_strategy=threshold_strategy,
        min_confidence_score=min_confidence_score,
    )
    results = _iter_object_detection_results(
        dataset_name,
        df,
        ground_truths_field=ground_truths_field,
        raw_inferences_field=raw_inferences_field,
        gt_ignore_property=gt_ignore_property,
        iou_threshold=iou_threshold,
        threshold_strategy=threshold_strategy,
        min_confidence_score=min_confidence_score,
        batch_size=batch_size,
        required_match_fields=required_match_fields,
    )
    dataset.upload_results(
        dataset_name,
        model_name,
        [(eval_config, results)],
        thresholded_fields=["thresholded"],
    )
