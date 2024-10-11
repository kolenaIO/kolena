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
from collections import defaultdict
from dataclasses import asdict
from typing import Any
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import pandas as pd

from kolena._api.v1.event import EventAPI
from kolena._api.v2._api import GeneralFieldFilter
from kolena._api.v2._filter import Filters
from kolena._api.v2._filter import ModelFilter
from kolena._api.v2._metric import Metric
from kolena._api.v2._testing import Path as TestingPath
from kolena._api.v2._testing import StratificationType
from kolena._api.v2._testing import StratifyFieldSpec
from kolena._api.v2._testing import TestingRequest
from kolena._api.v2._testing import TestingResponse
from kolena._api.v2.dataset import EntityData
from kolena._api.v2.model import ModelWithEvalConfig
from kolena._api.v2.quality_standard import CopyQualityStandardRequest
from kolena._api.v2.quality_standard import Path
from kolena._api.v2.quality_standard import QualityStandardResponse
from kolena._experimental.utils import get_delta_percentage
from kolena._experimental.utils import get_quantile
from kolena._experimental.utils import margin_of_error
from kolena._experimental.utils import ordinal
from kolena._utils import krequests_v2 as krequests
from kolena._utils import log
from kolena._utils.instrumentation import with_event
from kolena._utils.serde import from_dict
from kolena.dataset.dataset import _load_dataset_metadata
from kolena.dataset.evaluation import _get_eval_config_id
from kolena.dataset.evaluation import _get_model_id
from kolena.errors import IncorrectUsageError


PerformanceDelta = Literal["improved", "regressed", "similar", "unknown"]


def _format_quality_standard_result_df(quality_standard_result: dict) -> pd.DataFrame:
    df = pd.DataFrame.from_dict(quality_standard_result).rename_axis("stratification")

    df = df.explode("results_by_stratification")
    df = pd.concat(
        [df.drop(["results_by_stratification"], axis=1), df["results_by_stratification"].apply(pd.Series)],
        axis=1,
    )
    df.set_index(["test_case"], append=True, drop=True, inplace=True)

    df = df.explode("model_results")
    df["model"] = df["model_results"].apply(lambda x: x["model"]["name"])
    df["eval_config"] = df["model_results"].apply(lambda x: json.dumps(x["model"]["eval_config"]))
    df["metric_group"] = df["model_results"].apply(lambda x: list(x["results"].keys()))

    df = df.explode("metric_group")
    df["metric"] = df.apply(lambda x: list(x["model_results"]["results"][x["metric_group"]].keys()), axis=1)

    df = df.explode("metric")
    df["value"] = df.apply(lambda x: x["model_results"]["results"][x["metric_group"]][x["metric"]], axis=1)

    return df.pivot(columns=["model", "eval_config", "metric_group", "metric"], values="value")


def _download_quality_standard_result(
    dataset: str,
    models: List[str],
    metric_groups: Union[List[str], None] = None,
) -> pd.DataFrame:
    response = krequests.get(
        Path.RESULT,
        params=dict(dataset_name=dataset, models=models, metric_groups=metric_groups),
        api_version="v2",
    )
    krequests.raise_for_status(response)
    return _format_quality_standard_result_df(
        response.json(),
    )


def _download_test_case_result(
    dataset_id: int,
    model_and_eval_config_pairs: List[ModelWithEvalConfig],
    datapoint_filters: Optional[Dict[str, GeneralFieldFilter]],
    stratify_fields: List[StratifyFieldSpec],
) -> TestingResponse:
    filters = {}
    if datapoint_filters:
        for field, filter in datapoint_filters.items():
            if field.startswith("datapoint."):
                filters[field.replace("datapoint.", "")] = filter
            else:
                filters[field] = filter

    response = krequests.post(
        TestingPath.TESTING,
        json=asdict(
            TestingRequest(
                filters=Filters(
                    dataset_ids=[dataset_id],
                    datapoint=filters,
                    models=[
                        ModelFilter(
                            id=model.model_id,
                            eval_config_id=model.eval_config_id,
                        )
                        for model in model_and_eval_config_pairs
                    ],
                ),
                stratify_fields=stratify_fields,
            ),
        ),
        api_version="v2",
    )
    krequests.raise_for_status(response)
    return from_dict(TestingResponse, response.json())


def _download_quality_standard(
    dataset_id: int,
) -> QualityStandardResponse:
    response = krequests.get(
        Path.QUALITY_STANDARD,
        params=dict(dataset_id=dataset_id),
        api_version="v2",
    )
    krequests.raise_for_status(response)
    result = response.json()
    return from_dict(QualityStandardResponse, result)


def _calculate_moe_map(
    qs_result: pd.DataFrame,
    dataset_entity: EntityData,
    confidence_level: float,
    qs: QualityStandardResponse,
) -> Dict[Tuple[str, Any], float]:
    model_name_to_id_map = {}
    moe = {}
    for model, eval_config, _, _ in qs_result.columns:
        deserialized_eval_config = eval_config
        if eval_config == "null":
            deserialized_eval_config = None
        if eval_config and eval_config != "null":
            deserialized_eval_config = json.loads(eval_config)
        model_id = _get_model_id(model)
        eval_config_id = _get_eval_config_id(deserialized_eval_config)
        model_name_to_id_map[(model, eval_config)] = ModelWithEvalConfig(model_id, eval_config_id)

    for strat in qs.quality_standard.stratifications:
        test_cases = _download_test_case_result(
            dataset_entity.id,
            list(model_name_to_id_map.values()),
            strat.filters,
            strat.stratify_fields,
        )
        for i in range(len(test_cases.test_cases)):
            case = test_cases.test_cases[i]
            if not case.stratification or not case.stratification[0]:
                continue
            value = case.stratification[0].value
            field = case.stratification[0].field
            if field.startswith("_kolena.extracted"):
                field_keys = field.split(".")
                field = ".".join(field_keys[3:])
            if case.stratification[0].type in {StratificationType.EQUAL_HEIGHT, StratificationType.EQUAL_WIDTH}:
                value = f"{ordinal(i + 1)} {get_quantile(len(test_cases.test_cases))}"
            moe[(field, value)] = margin_of_error(case.sample_count, confidence_level)
        if ("Dataset", np.nan) not in moe:
            moe[("Dataset", np.nan)] = margin_of_error(test_cases.dataset.sample_count, confidence_level)
    return moe


def _get_performance_delta_metrics(
    qs: QualityStandardResponse,
    qs_result: pd.DataFrame,
    moe: Dict[Tuple[str, Any], float],
    reference_model: Optional[str] = None,
    reference_eval_config: Union[Dict[str, Any], str, None] = "first",
) -> pd.DataFrame:
    if not reference_eval_config:
        reference_eval_config = "null"
    for model, eval_config, _, _ in qs_result.columns:
        if not reference_model:
            reference_model = model
        if reference_model == model and reference_eval_config == "first":
            reference_eval_config = eval_config
    if isinstance(reference_eval_config, dict):
        reference_eval_config = json.dumps(reference_eval_config)
    performance_delta = pd.DataFrame("unknown", index=qs_result.index, columns=qs_result.columns)
    metric_dict: Dict[str, Dict[str, Metric]] = defaultdict(dict)
    for mg in qs.quality_standard.metric_groups:
        for metric in mg.metrics:
            metric_dict[mg.name][metric.label] = metric

    for model, eval_config, metric_group, metric_name in qs_result.columns:
        highlight = metric_dict[metric_group][metric_name].highlight
        if not highlight or highlight.higherIsBetter is None:
            continue

        for strat, test_case in qs_result.index:
            ref_val = qs_result.loc[
                (strat, test_case),
                (
                    reference_model,
                    reference_eval_config,
                    metric_group,
                    metric_name,
                ),
            ]
            target_val = qs_result.loc[(strat, test_case), (model, eval_config, metric_group, metric_name)]
            if (
                ref_val is None
                or target_val is None
                or np.isnan(ref_val)
                or np.isnan(target_val)
                or (strat, test_case) not in moe
            ):
                continue
            range_max = np.nanmax(
                qs_result.loc[strat][(reference_model, reference_eval_config, metric_group, metric_name)],
            )
            delta_percentage = get_delta_percentage(
                target_val,
                ref_val,
                range_max if range_max and not np.isnan(range_max) else max(target_val, ref_val),
            )
            if not highlight.higherIsBetter:
                delta_percentage = -delta_percentage
            if delta_percentage > moe[(strat, test_case)]:
                performance_delta.loc[
                    (strat, test_case),
                    (model, eval_config, metric_group, metric_name),
                ] = "improved"
            elif delta_percentage < -1 * moe[(strat, test_case)]:  # noqa: PAR001
                performance_delta.loc[
                    (strat, test_case),
                    (model, eval_config, metric_group, metric_name),
                ] = "regressed"
            else:
                performance_delta.loc[
                    (strat, test_case),
                    (model, eval_config, metric_group, metric_name),
                ] = "similar"
    return performance_delta


def _calculate_performance_delta(
    dataset: str,
    qs_result: pd.DataFrame,
    confidence_level: float,
    reference_model: Optional[str] = None,
    reference_eval_config: Union[Dict[str, Any], Literal["first"], None] = "first",
) -> pd.DataFrame:
    dataset_entity = _load_dataset_metadata(dataset)
    if not dataset_entity:
        raise IncorrectUsageError(f"The dataset with name '{dataset}' not found")
    qs = _download_quality_standard(dataset_entity.id)
    moe = _calculate_moe_map(qs_result, dataset_entity, confidence_level, qs)
    performance_delta = _get_performance_delta_metrics(qs, qs_result, moe, reference_model, reference_eval_config)

    qs_result = pd.concat(
        [qs_result, performance_delta],
        axis=1,
        keys=["metric_value", "performance_delta"],
        names=["type"],
    )
    qs_result = qs_result.reorder_levels(["model", "eval_config", "metric_group", "metric", "type"], axis=1)
    return qs_result


@with_event(event_name=EventAPI.Event.FETCH_QUALITY_STANDARD_RESULT)
def download_quality_standard_result(
    dataset: str,
    models: List[str],
    metric_groups: Union[List[str], None] = None,
    intersect_results: bool = True,
    confidence_level: Optional[float] = None,
    reference_model: Optional[str] = None,
    reference_eval_config: Optional[Union[Dict[str, Any], Literal["first"]]] = "first",
) -> pd.DataFrame:
    """
    Download quality standard result given a dataset and list of models.

    :param dataset: The name of the dataset.
    :param models: The names of the models.
    :param metric_groups: The names of the metric groups to include in the result.
    :param intersect_results: If True, only include datapoint that are common to all models in the metrics calculation.
    Note all metric groups are included when this value is `None`.
    :param confidence_level: The confidence score used to calculate the Margin of Error, representing the probability of
     capturing the true population parameter within the calculated MOE, we recommend setting between 0.9 - 0.99.
     If this is specified performance delta column will be shown, metric valued will be classified as
      improved, regressed, similar or unknown according to the MOE.
    :param reference_model: The name of the model to use as a reference for the Margin of Error calculation.
        This should be one of the models provided in `models`. If unspecified, the first model of `models` will be used.
    :param reference_eval_config: The evaluation configuration to use in conjunction with the reference model,
     if unspecified the first evaluation configuration of the reference model will be used
    :return: A Dataframe containing the quality standard result.
    """
    if reference_model and reference_model not in models:
        raise IncorrectUsageError(
            f"The specified reference model '{reference_model}' was not one of the provided models",
        )
    model_log = ", ".join([f"'{model}'" for model in models])
    log.info(f"downloading quality standard results for model(s) {model_log} on dataset '{dataset}'")
    if intersect_results:
        qs_result = _download_quality_standard_result(dataset, models, metric_groups)
    else:
        result_dfs = []
        for model in models:
            result_dfs.append(_download_quality_standard_result(dataset, [model], metric_groups))
        qs_result = pd.concat(result_dfs, axis=1)

    if confidence_level:
        qs_result = _calculate_performance_delta(
            dataset,
            qs_result,
            confidence_level,
            reference_model,
            reference_eval_config,
        )
    return qs_result


@with_event(event_name=EventAPI.Event.COPY_QUALITY_STANDARD_FROM_DATASET)
def copy_quality_standards_from_dataset(
    dataset: str,
    source_dataset: str,
    include_metric_groups: bool = True,
    include_test_cases: bool = True,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Create a quality standard on a dataset by copying from a source dataset. Note that this operation will overwrite the
    existing quality standards on the dataset if they exist.

    :param dataset: The name of the dataset.
    :param source_dataset: The name of the dataset from which the quality standards should be copied.
    :param include_metric_groups: Optional flag to indicate whether to copy the metric groups from the source dataset.
    :param include_test_cases: Optional flag to indicate whether to copy the test cases from the source dataset.
    :return: A tuple of the created metric groups and test cases.
    """
    if dataset == source_dataset:
        raise IncorrectUsageError("source dataset and target dataset are the same")

    if not include_test_cases and not include_metric_groups:
        raise IncorrectUsageError("should include at least one of metric group or test case")

    dataset_metadata = _load_dataset_metadata(dataset)
    if not dataset_metadata:
        raise IncorrectUsageError(f"The dataset with name '{dataset}' not found")
    source_dataset_metadata = _load_dataset_metadata(source_dataset)
    if not source_dataset_metadata:
        raise IncorrectUsageError(f"The source dataset with name '{source_dataset}' not found")

    request = CopyQualityStandardRequest(
        dataset_metadata.id,
        source_dataset_metadata.id,
        include_metric_groups=include_metric_groups,
        include_test_cases=include_test_cases,
    )

    response = krequests.put(
        Path.COPY_FROM_DATASET,
        json=asdict(request),
        api_version="v2",
    )
    krequests.raise_for_status(response)

    metric_groups = response.json().get("metric_groups", [])
    test_cases = response.json().get("stratifications", [])
    return metric_groups, test_cases
