# Copyright 2021-2025 Kolena Inc.
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
from typing import Any
from typing import Dict
from typing import Iterable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
import tqdm

from kolena import dataset
from kolena._experimental.classification._matching import InferenceMatches
from kolena._experimental.classification._matching import match_inferences
from kolena.annotation import Label
from kolena.annotation import ScoredLabel
from kolena.dataset.dataset import _load_dataset_metadata
from kolena.errors import IncorrectUsageError


def _label_as_dict(raw_label: Union[str, Label, ScoredLabel]) -> Dict[str, Any]:
    if isinstance(raw_label, str):
        return dict(label=raw_label)
    return raw_label._to_dict()


def _merge_labels(
    gt: Union[str, Label, ScoredLabel],
    inf: Union[str, Label, ScoredLabel],
) -> Union[str, Dict[str, Any]]:
    if isinstance(gt, str) and isinstance(inf, str):
        return gt
    return {**_label_as_dict(gt), **_label_as_dict(inf)}


def _datapoint_metrics(
    object_matches: InferenceMatches,
) -> Dict[str, Any]:
    tp = [_merge_labels(gt, inf) for gt, inf in object_matches.matched]
    fp = object_matches.unmatched_inf
    fn = object_matches.unmatched_gt
    count_tp = len(tp)
    count_fp = len(fp)
    count_fn = len(fn)
    precision = count_tp / (count_tp + count_fp) if count_tp + count_fp > 0 else 0
    recall = count_tp / (count_tp + count_fn) if count_tp + count_fn > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
    jaccard_index = count_tp / (count_tp + count_fp + count_fn) if count_tp + count_fp + count_fn > 0 else 0

    return dict(
        TP=tp,
        FP=fp,
        FN=fn,
        count_TP=len(tp),
        count_FP=len(fp),
        count_FN=len(fn),
        is_exact_match=len(fp) + len(fn) == 0,
        Precision=precision,
        Recall=recall,
        F1_Score=f1,
        Jaccard_Index=jaccard_index,
    )


def _iter_metrics(
    pred_df: pd.DataFrame,
    all_object_matches: List[InferenceMatches],
    *,
    batch_size: int = 10_000,
) -> Iterator[pd.DataFrame]:
    for i in tqdm.tqdm(range(0, pred_df.shape[0], batch_size)):
        metrics = [_datapoint_metrics(matches) for matches in all_object_matches[i : i + batch_size]]
        pred_df = pred_df.reset_index(drop=True)
        pred_df["multilabel_classification.metrics"] = metrics
        yield pred_df


def _compute_metrics(
    pred_df: pd.DataFrame,
    *,
    ground_truth: str,
    inference: str,
    gt_ignore_property: Optional[str] = None,
    batch_size: int = 10_000,
    required_match_fields: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    """
    Compute metrics for multilabel classification.

    :param df: Dataframe for model results.
    :param ground_truth: Column name for ground truth object labels
    :param inference: Column name for inference object labels
    :param gt_ignore_property: Field on the ground truth labels used to determine if the label should be
    ignored. Labels will be ignored if this field exists and is equal to `True`.
    :param batch_size: number of results to process per iteration.
    :param Optional[List[str]] required_match_fields: Optionally specify a list of fields that must match between
        the inference and ground truth for them to be considered a match. This only applies if both the ground truth
        and inference fields are lists of [`Labels`][kolena.annotation.Label].
    """
    idx = {name: i for i, name in enumerate(list(pred_df), start=1)}

    all_object_matches: List[InferenceMatches] = []
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
            match_inferences(
                unignored_ground_truths,
                inferences,
                ignored_ground_truths=ignored_ground_truths,
                required_match_fields=required_match_fields,
            ),
        )

    pred_df.drop(columns=ground_truth, inplace=True)
    yield from _iter_metrics(
        pred_df,
        all_object_matches,
        batch_size=batch_size,
    )


def _safe_get_label(obj: object) -> Optional[str]:
    if isinstance(obj, str):
        return obj
    if hasattr(obj, "label"):
        return str(obj.label)

    return None


def _safe_get_scored(obj: object) -> Optional[bool]:
    return hasattr(obj, "score")


def _get_labels_from_objects(objs: Iterable[object]) -> List[str]:
    maybe_labels = {_safe_get_label(obj) for obj in objs}
    labels = [label for label in maybe_labels if label is not None]
    return labels


def _check_scored(inference: pd.Series) -> bool:
    scored = {hasattr(inf, "score") for inf in itertools.chain.from_iterable(_filter_null(inference))}
    return len(scored) == 1 and (True in scored)


def _filter_null(series: pd.Series) -> pd.Series:
    return series[series.notnull()]


def _validate_column_present(df: pd.DataFrame, col: str) -> None:
    if col not in df.columns:
        raise IncorrectUsageError(f"Missing column '{col}'")


def _iter_multilabel_classification_results(
    dataset_name: str,
    df: pd.DataFrame,
    *,
    ground_truths_field: str = "ground_truths",
    raw_inferences_field: str = "raw_inferences",
    gt_ignore_property: Optional[str] = None,
    batch_size: int = 10_000,
    required_match_fields: Optional[List[str]] = None,
) -> Iterator[pd.DataFrame]:
    _validate_column_present(df, raw_inferences_field)

    dataset_metadata = _load_dataset_metadata(dataset_name)
    if dataset_metadata is None:
        raise RuntimeError("error retrieving dataset id fields")
    id_fields = dataset_metadata.id_fields
    dataset_df = dataset.download_dataset(dataset_name)
    dataset_df = dataset_df[[*id_fields, ground_truths_field]]
    _validate_column_present(dataset_df, ground_truths_field)
    while ground_truths_field in df.columns:
        new_ground_truths_field = f"_kolena.rename.{ground_truths_field}"
        dataset_df = dataset_df.rename(columns={ground_truths_field: new_ground_truths_field})
        ground_truths_field = new_ground_truths_field

    merged_df = df.merge(dataset_df, on=id_fields)
    return _compute_metrics(
        merged_df,
        ground_truth=ground_truths_field,
        inference=raw_inferences_field,
        gt_ignore_property=gt_ignore_property,
        batch_size=batch_size,
        required_match_fields=required_match_fields,
    )


def upload_multilabel_classification_results(
    dataset_name: str,
    model_name: str,
    df: pd.DataFrame,
    *,
    ground_truths_field: str = "ground_truths",
    raw_inferences_field: str = "raw_inferences",
    gt_ignore_property: Optional[str] = None,
    batch_size: int = 10_000,
    required_match_fields: Optional[List[str]] = None,
) -> None:
    """
    Compute metrics and upload results of the model for the dataset.

    Dataframe `df` should include all id columns that would match to that of corresponding datapoint and
    an `inference` column that should be a list of either `str` or scored / un-scored
    [`Labels`][kolena.annotation.Label].

    :param dataset_name: Dataset name.
    :param model_name: Model name.
    :param df: Dataframe for model results.
    :param ground_truths_field: Field name in datapoint with ground truth labels,
    defaulting to `"ground_truths"`.
    :param raw_inferences_field: Column in model result DataFrame with raw inference labels,
    defaulting to `"raw_inferences"`. These inferences will be directly matched against the ground truths, and
    should be pre-filtered for any disqualifying factors, such as confidence.
    :param gt_ignore_property: Name of a property on the ground truth labels used to determine if the label
    should be ignored. Labels will be ignored if this property exists and is equal to `True`.
    :param batch_size: number of results to process per iteration.
    :param Optional[List[str]] required_match_fields: Optionally specify a list of fields that must match between
        the inference and ground truth for them to be considered a match. This only applies if both the ground truth
        and inference fields are lists of [`Labels`][kolena.annotation.Label].
    :return:
    """

    results = _iter_multilabel_classification_results(
        dataset_name,
        df,
        ground_truths_field=ground_truths_field,
        raw_inferences_field=raw_inferences_field,
        gt_ignore_property=gt_ignore_property,
        batch_size=batch_size,
        required_match_fields=required_match_fields,
    )
    dataset.upload_results(
        dataset_name,
        model_name,
        results,
    )
