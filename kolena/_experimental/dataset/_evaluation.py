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
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd
from pandas.errors import MergeError

from kolena._api.v2.model import LoadResultsRequest
from kolena._api.v2.model import Path
from kolena._api.v2.model import UploadResultsRequest
from kolena._experimental.dataset._dataset import _iter_dataset_raw
from kolena._experimental.dataset._dataset import _to_deserialized_dataframe
from kolena._experimental.dataset._dataset import _to_serialized_dataframe
from kolena._experimental.dataset.common import COL_DATAPOINT
from kolena._experimental.dataset.common import COL_EVAL_CONFIG
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
TEST_ON_TYPE = Optional[Union[str, List[str]]]


def _fetch_dataset(dataset: str) -> pd.DataFrame:
    df_data_batch = list(_iter_dataset_raw(dataset))
    df_datapoints = pd.concat(df_data_batch) if df_data_batch else pd.DataFrame(columns=["id", COL_DATAPOINT])
    df_datapoints.rename(columns={"id": "datapoint_id"}, inplace=True)
    return df_datapoints


def _iter_result_raw(dataset: str, model: str, batch_size: int) -> Iterator[pd.DataFrame]:
    validate_batch_size(batch_size)
    init_request = LoadResultsRequest(dataset=dataset, model=model, batch_size=batch_size)
    yield from _BatchedLoader.iter_data(
        init_request=init_request,
        endpoint_path=Path.LOAD_RESULTS.value,
        df_class=None,
        endpoint_api_version=API_V2,
    )


def _fetch_results(dataset: str, model: str) -> pd.DataFrame:
    df_result_batch = list(_iter_result_raw(dataset, model, batch_size=BatchSize.LOAD_RECORDS))
    return (
        pd.concat(df_result_batch)
        if df_result_batch
        else pd.DataFrame(
            columns=["datapoint_id", COL_DATAPOINT, COL_RESULT, COL_EVAL_CONFIG],
        )
    )


def _drop_unprovided_result(
    df_result_concat: pd.DataFrame,
    df_result_input: pd.DataFrame,
    on: TEST_ON_TYPE,
) -> pd.DataFrame:
    if not on:
        return df_result_concat

    if isinstance(on, str):
        on = [on]

    _validate_on(df_result_concat, df_result_input, on)
    df_result_provided = df_result_input[on].merge(df_result_concat, how="inner", on=on)
    return df_result_provided


def _process_results(
    df: pd.DataFrame,
    all_results: List[Tuple[Optional[TYPE_EVALUATION_CONFIG], pd.DataFrame, pd.DataFrame]],
    df_datapoints: pd.DataFrame,
    on: TEST_ON_TYPE,
) -> pd.DataFrame:
    target_columns = ["datapoint_id", COL_RESULT, COL_EVAL_CONFIG]
    df_result_by_eval = []
    for eval_config, df_result, df_result_input in all_results:
        df_result_eval = _to_serialized_dataframe(df_result, column=COL_RESULT)
        df_result_eval[COL_EVAL_CONFIG] = json.dumps(eval_config) if eval_config is not None else None

        df_result_concat = pd.concat([df["datapoint_id"], df_datapoints, df_result_eval], axis=1)
        df_result_concat = _drop_unprovided_result(df_result_concat, df_result_input, on)

        df_result_by_eval.append(df_result_concat[target_columns])
    df_results = (
        pd.concat(df_result_by_eval, ignore_index=True)
        if df_result_by_eval
        else pd.DataFrame(
            columns=target_columns,
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
    df = _fetch_results(dataset, model)

    df_datapoints = _to_deserialized_dataframe(df.drop_duplicates(subset=[COL_DATAPOINT]), column=COL_DATAPOINT)
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


def _validate_configs(configs: List[TYPE_EVALUATION_CONFIG]) -> None:
    n = len(configs)
    for i in range(n):
        for j in range(i + 1, n):
            if configs[i] == configs[j]:
                raise IncorrectUsageError("duplicate eval configs are invalid")


def _validate_data(left: pd.DataFrame, right: pd.DataFrame) -> None:
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


def _align_datapoints_results(
    df_datapoints: pd.DataFrame,
    df_result_input: pd.DataFrame,
    on: TEST_ON_TYPE,
) -> pd.DataFrame:
    if not on:
        return df_result_input

    if isinstance(on, str):
        on = [on]

    _validate_on(df_datapoints, df_result_input, on)
    try:
        df_result = df_datapoints[on].merge(df_result_input, how="left", on=on, validate="one_to_one")
    except MergeError as e:
        raise IncorrectUsageError(f"merge key {on} is not unique") from e

    return df_result.drop(columns=on)


def test(
    dataset: str,
    model: str,
    results: Union[pd.DataFrame, List[Tuple[TYPE_EVALUATION_CONFIG, pd.DataFrame]]],
    on: TEST_ON_TYPE = None,
) -> None:
    """
    This function is used for testing a specified model on a given dataset.

    :param dataset: The name of the dataset to be used.
    :param model: The name of the model to be used.
    :param results: Either a DataFrame or a list of tuples, where each tuple consists of
                    a eval configuration and a DataFrame.
    :param on: The column(s) to merge on between datapoint DataFrame and result DataFrame

    :return None
    """
    df_data = _fetch_dataset(dataset)
    df_datapoints = _to_deserialized_dataframe(df_data, column=COL_DATAPOINT)
    log.info(f"fetched {len(df_data)} for dataset {dataset}")

    if isinstance(results, pd.DataFrame):
        results = [(None, results)]

    _validate_configs([cfg for cfg, _ in results])

    all_results: List[Tuple[Optional[TYPE_EVALUATION_CONFIG], pd.DataFrame, pd.DataFrame]] = []
    for config, df_result_input in results:
        log.info(f"start evaluation with configuration {config}" if config else "start evaluation")
        single_result = _align_datapoints_results(df_datapoints, df_result_input, on)
        _validate_data(df_datapoints, single_result)
        all_results.append((config, single_result, df_result_input))
        log.info(f"completed evaluation with configuration {config}" if config else "completed evaluation")

    df_results = _process_results(df_data, all_results, df_datapoints, on)
    n_uploaded_results = _upload_results(model, df_results)
    log.info(f"uploaded {n_uploaded_results} test results")
