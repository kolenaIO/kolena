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
import json
from dataclasses import asdict
from typing import Any
from typing import Callable
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd

from kolena._api.v2.evaluation import LoadMetricsRequest
from kolena._api.v2.evaluation import Path as EvaluationPath
from kolena._api.v2.evaluation import UploadMetricsRequest
from kolena._api.v2.model import LoadInferencesRequest
from kolena._api.v2.model import LoadResultsRequest
from kolena._api.v2.model import Path
from kolena._api.v2.model import UploadInferencesRequest
from kolena._api.v2.model import UploadResultsRequest
from kolena._experimental.dataset._dataset import _iter_dataset_raw
from kolena._experimental.dataset._dataset import _to_deserialized_dataframe
from kolena._experimental.dataset._dataset import _to_serialized_dataframe
from kolena._experimental.dataset.common import COL_DATAPOINT
from kolena._experimental.dataset.common import COL_EVAL_CONFIG
from kolena._experimental.dataset.common import COL_INFERENCE
from kolena._experimental.dataset.common import COL_METRICS
from kolena._experimental.dataset.common import COL_RESULT
from kolena._experimental.dataset.common import validate_batch_size
from kolena._utils import krequests_v2 as krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.state import API_V2
from kolena.errors import IncorrectUsageError


TYPE_EVALUATION_CONFIG = Optional[Dict[str, Any]]
INFER_FUNC_TYPE = Callable[[pd.DataFrame], pd.DataFrame]
EVAL_FUNC_TYPE = Callable[[pd.DataFrame, pd.DataFrame, Optional[TYPE_EVALUATION_CONFIG]], pd.DataFrame]
TEST_INFER_TYPE = Optional[Union[INFER_FUNC_TYPE, pd.DataFrame]]
TEST_EVAL_TYPE = Optional[Union[EVAL_FUNC_TYPE, pd.DataFrame]]
TEST_EVAL_CONFIGS_TYPE = Optional[Union[TYPE_EVALUATION_CONFIG, List[TYPE_EVALUATION_CONFIG]]]
TEST_ON_TYPE = Optional[Union[str, List[str]]]


def _fetch_dataset(dataset: str) -> pd.DataFrame:
    df_data_batch = list(_iter_dataset_raw(dataset))
    return pd.concat(df_data_batch) if df_data_batch else pd.DataFrame(columns=["id", COL_DATAPOINT])


def _iter_inference_raw(dataset: str, model: str, batch_size: int) -> Iterator[pd.DataFrame]:
    validate_batch_size(batch_size)
    init_request = LoadInferencesRequest(dataset=dataset, model=model, batch_size=batch_size)
    yield from _BatchedLoader.iter_data(
        init_request=init_request,
        endpoint_path=Path.LOAD_INFERENCES.value,
        df_class=None,
        endpoint_api_version=API_V2,
    )


def _iter_metrics_raw(dataset: str, model: str, batch_size: int) -> Iterator[pd.DataFrame]:
    validate_batch_size(batch_size)
    init_request = LoadMetricsRequest(dataset=dataset, model=model, batch_size=batch_size)
    yield from _BatchedLoader.iter_data(
        init_request=init_request,
        endpoint_path=EvaluationPath.LOAD_METRICS.value,
        df_class=None,
        endpoint_api_version=API_V2,
    )


def _iter_result_raw(dataset: str, model: str, batch_size: int) -> Iterator[pd.DataFrame]:
    validate_batch_size(batch_size)
    init_request = LoadResultsRequest(dataset=dataset, model=model, batch_size=batch_size)
    yield from _BatchedLoader.iter_data(
        init_request=init_request,
        endpoint_path=Path.LOAD_RESULTS.value,
        df_class=None,
        endpoint_api_version=API_V2,
    )


def _fetch_inferences(dataset: str, model: str) -> pd.DataFrame:
    df_inference_batch = list(_iter_inference_raw(dataset, model, batch_size=BatchSize.LOAD_RECORDS))
    return (
        pd.concat(df_inference_batch)
        if df_inference_batch
        else pd.DataFrame(
            columns=["datapoint_id", COL_DATAPOINT, "inference_id", COL_INFERENCE],
        )
    )


def fetch_inferences(dataset: str, model: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetch inferences given dataset name and model name.
    """
    df_data = _fetch_inferences(dataset, model)
    df_datapoints = _to_deserialized_dataframe(df_data, column=COL_DATAPOINT)
    df_inferences = _to_deserialized_dataframe(df_data, column=COL_INFERENCE)

    return df_datapoints, df_inferences


def _fetch_results(dataset: str, model: str) -> pd.DataFrame:
    df_result_batch = list(_iter_result_raw(dataset, model, batch_size=BatchSize.LOAD_RECORDS))
    return (
        pd.concat(df_result_batch)
        if df_result_batch
        else pd.DataFrame(
            columns=["datapoint_id", COL_DATAPOINT, COL_RESULT, COL_EVAL_CONFIG],
        )
    )


def _upload_inferences(model: str, df: pd.DataFrame) -> None:
    load_uuid = init_upload().uuid
    upload_data_frame(
        df=df[["id", COL_INFERENCE]],
        batch_size=BatchSize.UPLOAD_RECORDS.value,
        load_uuid=load_uuid,
    )
    request = UploadInferencesRequest(model=model, uuid=load_uuid)
    response = krequests.post(Path.UPLOAD_INFERENCES, json=asdict(request))
    krequests.raise_for_status(response)


def _process_metrics(
    df: pd.DataFrame,
    all_metrics: List[Tuple[Optional[TYPE_EVALUATION_CONFIG], pd.DataFrame]],
) -> pd.DataFrame:
    df_metrics_by_eval = []
    for eval_config, df_metrics in all_metrics:
        df_metrics_eval = _to_serialized_dataframe(df_metrics, column=COL_METRICS)
        df_metrics_eval[COL_EVAL_CONFIG] = json.dumps(eval_config) if eval_config is not None else None
        df_metrics_by_eval.append(pd.concat([df["inference_id"], df_metrics_eval], axis=1))
    df_metrics = (
        pd.concat(df_metrics_by_eval, ignore_index=True)
        if df_metrics_by_eval
        else pd.DataFrame(
            columns=["inference_id", COL_METRICS],
        )
    )
    return df_metrics


def _upload_metrics(
    df: pd.DataFrame,
) -> int:
    if df["inference_id"].isnull().any():
        raise IncorrectUsageError("cannot upload metrics without inference")

    load_uuid = init_upload().uuid
    upload_data_frame(df=df, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=load_uuid)
    request = UploadMetricsRequest(uuid=load_uuid)
    response = krequests.post(EvaluationPath.UPLOAD_METRICS, json=asdict(request))
    krequests.raise_for_status(response)
    return len(df)


def _process_results(
    df: pd.DataFrame,
    all_results: List[Tuple[Optional[TYPE_EVALUATION_CONFIG], pd.DataFrame]],
) -> pd.DataFrame:
    df_results_by_eval = []
    for eval_config, df_results in all_results:
        df_results_eval = _to_serialized_dataframe(df_results, column=COL_RESULT)
        df_results_eval[COL_EVAL_CONFIG] = json.dumps(eval_config) if eval_config is not None else None
        df_results_by_eval.append(pd.concat([df["datapoint_id"], df_results_eval], axis=1))
    df_results = (
        pd.concat(df_results_by_eval, ignore_index=True)
        if df_results_by_eval
        else pd.DataFrame(
            columns=["datapoint_id", COL_RESULT, COL_EVAL_CONFIG],
        )
    )
    return df_results


def _upload_results(
    model: str,
    df: pd.DataFrame,
) -> int:
    load_uuid = init_upload().uuid
    upload_data_frame(df=df, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=load_uuid)
    request = UploadResultsRequest(model=model, uuid=load_uuid)
    response = krequests.post(Path.UPLOAD_RESULTS, json=asdict(request))
    krequests.raise_for_status(response)
    return len(df)


def fetch_results(
    dataset: str,
    model: str,
) -> Tuple[pd.DataFrame, List[Tuple[TYPE_EVALUATION_CONFIG, pd.DataFrame]]]:
    """
    Fetch results given dataset name and model name.
    """
    df_results_batch = list(_iter_result_raw(dataset, model, batch_size=BatchSize.LOAD_RECORDS))
    df = (
        pd.concat(df_results_batch)
        if df_results_batch
        else pd.DataFrame(
            columns=[COL_DATAPOINT, COL_RESULT, COL_EVAL_CONFIG],
        )
    )

    df_datapoints = _to_deserialized_dataframe(df, column=COL_DATAPOINT)
    eval_configs = df[COL_EVAL_CONFIG].unique()
    df_results_by_eval = []
    for eval_config in eval_configs:
        df_matched = df[df[COL_EVAL_CONFIG] == eval_config if eval_config is not None else df[COL_EVAL_CONFIG].isnull()]
        df_results_by_eval.append(
            (
                json.loads(eval_config) if eval_config is not None else None,
                _to_deserialized_dataframe(df_matched, column=COL_RESULT),
            ),
        )

    return df_datapoints, df_results_by_eval


def fetch_evaluation_results(
    dataset: str,
    model: str,
) -> List[Tuple[Dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Fetch evaluation results given dataset name and model name.
    """
    df_results_batch = list(_iter_metrics_raw(dataset, model, batch_size=BatchSize.LOAD_RECORDS))
    df = (
        pd.concat(df_results_batch)
        if df_results_batch
        else pd.DataFrame(
            columns=[COL_DATAPOINT, COL_INFERENCE, COL_METRICS, COL_EVAL_CONFIG],
        )
    )

    eval_configs = df[COL_EVAL_CONFIG].unique()
    df_by_eval = []
    for eval_config in eval_configs:
        df_matched = df[df[COL_EVAL_CONFIG] == eval_config if eval_config is not None else df[COL_EVAL_CONFIG].isnull()]
        df_by_eval.append(
            (
                json.loads(eval_config) if eval_config is not None else None,
                _to_deserialized_dataframe(df_matched, column=COL_DATAPOINT),
                _to_deserialized_dataframe(df_matched, column=COL_INFERENCE),
                _to_deserialized_dataframe(df_matched, column=COL_METRICS),
            ),
        )

    return df_by_eval


def validate_data(left: pd.DataFrame, right: pd.DataFrame) -> None:
    if len(left) != len(right):
        raise IncorrectUsageError("numbers of rows between two dataframe do not match")


def _validate_on(left: pd.DataFrame, right: pd.DataFrame, on: TEST_ON_TYPE) -> None:
    if on is None:
        raise IncorrectUsageError("on cannot be None")
    # works for both string and list of string
    if len(on) == 0:
        raise IncorrectUsageError("on cannot be empty")

    if isinstance(on, str):
        if on not in left.columns or on not in right.columns:
            raise IncorrectUsageError(f"column {on} doesn't exist in target dataframe")
    elif isinstance(on, list):
        for col in on:
            if col not in left.columns or col not in right.columns:
                raise IncorrectUsageError(f"column {col} doesn't exist in target dataframe")


def _get_default_infer_func(df_inf: pd.DataFrame, on: TEST_ON_TYPE) -> INFER_FUNC_TYPE:
    if isinstance(on, str):
        on = [on]

    def infer(datapoints: pd.DataFrame) -> pd.DataFrame:
        inferences = datapoints[on].merge(df_inf, how="left", on=on)
        return inferences

    return infer


def _get_default_eval_func(df_metrics: pd.DataFrame, on: TEST_ON_TYPE) -> EVAL_FUNC_TYPE:
    if isinstance(on, str):
        on = [on]

    def eval(
        datapoints: pd.DataFrame,
        inferences: pd.DataFrame,
        eval_config: Optional[TYPE_EVALUATION_CONFIG],
    ) -> pd.DataFrame:
        metrics = datapoints[on].merge(df_metrics, how="left", on=on)
        return metrics

    return eval


def _get_df_inferences(infer: TEST_INFER_TYPE, df_datapoints: pd.DataFrame, on: TEST_ON_TYPE) -> pd.DataFrame:
    if isinstance(infer, pd.DataFrame):
        _validate_on(df_datapoints, infer, on)
        infer = _get_default_infer_func(infer, on)

    df_inferences = infer(df_datapoints)
    return df_inferences


def _get_single_df_metrics(
    eval: TEST_INFER_TYPE,
    df_datapoints: pd.DataFrame,
    df_inferences: pd.DataFrame,
    config: Optional[TYPE_EVALUATION_CONFIG],
    on: TEST_ON_TYPE,
) -> pd.DataFrame:
    if isinstance(eval, pd.DataFrame):
        _validate_on(df_datapoints, eval, on)
        eval = _get_default_eval_func(eval, on)

    single_metrics = eval(df_datapoints, df_inferences, config)
    return single_metrics


def _get_single_df_result(
    df_datapoints: pd.DataFrame,
    df_result_input: pd.DataFrame,
    on: TEST_ON_TYPE,
) -> pd.DataFrame:
    if not on:
        return df_result_input

    _validate_on(df_datapoints, df_result_input, on)
    df_result = df_datapoints[on].merge(df_result_input, how="left", on=on)
    return df_result


def test(
    dataset: str,
    model: str,
    results: Union[pd.DataFrame, List[Tuple[TYPE_EVALUATION_CONFIG, pd.DataFrame]]],
    on: TEST_ON_TYPE = None,
) -> None:
    """
    # TODO: docstring
    This function is used for testing on a given dataset using a specified model.

    :param dataset: The name of the dataset to be used.
    :param model: The name of the model to be used.
    :param results: ...
    :param on: ...

    :return None: This function doesn't return anything.
    """
    df_data = _fetch_dataset(dataset)
    df_datapoints = _to_deserialized_dataframe(df_data, column=COL_DATAPOINT)
    log.info(f"fetched {len(df_data)} for dataset {dataset}")

    if isinstance(results, pd.DataFrame):
        results = [(None, results)]

    all_results = []
    for config, df_result_input in results:
        log.info(f"start test with configuration {config}" if config else "start evaluation")
        single_result = _get_single_df_result(df_datapoints, df_result_input, on)
        validate_data(df_datapoints, single_result)
        all_results.append((config, single_result))
        log.info(f"completed test with configuration {config}" if config else "completed evaluation")

    df_results = _process_results(df_datapoints, all_results)
    n_uploaded_results = _upload_results(model, df_results)
    log.info(f"uploaded {n_uploaded_results} test results")
