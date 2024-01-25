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
from dataclasses import asdict
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pandas as pd

from kolena._api.v1.event import EventAPI
from kolena._api.v2.model import LoadResultsRequest
from kolena._api.v2.model import Path
from kolena._api.v2.model import UploadResultsRequest
from kolena._api.v2.model import UploadResultsResponse
from kolena._utils import krequests_v2 as krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.instrumentation import with_event
from kolena._utils.serde import from_dict
from kolena._utils.state import API_V2
from kolena.dataset._common import COL_DATAPOINT
from kolena.dataset._common import COL_DATAPOINT_ID_OBJECT
from kolena.dataset._common import COL_EVAL_CONFIG
from kolena.dataset._common import COL_RESULT
from kolena.dataset._common import DEFAULT_SOURCES
from kolena.dataset._common import validate_batch_size
from kolena.dataset._common import validate_dataframe_have_other_columns_besides_ids
from kolena.dataset._common import validate_dataframe_ids
from kolena.dataset.dataset import _load_dataset_metadata
from kolena.dataset.dataset import _to_deserialized_dataframe
from kolena.dataset.dataset import _to_serialized_dataframe
from kolena.errors import IncorrectUsageError
from kolena.errors import NotFoundError

EvalConfig = Optional[Dict[str, Any]]
"""
User defined configuration for evaluating results, for example `{"threshold": 7}`.
"""
DataFrame = Union[pd.DataFrame, Iterator[pd.DataFrame]]
"""
A type alias representing a DataFrame, which can be either a pandas DataFrame
or an iterator of pandas DataFrames.
"""


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


def _process_result(
    eval_config: EvalConfig,
    df_result: pd.DataFrame,
    id_fields: List[str],
) -> pd.DataFrame:
    df_serialized_datapoint_id_object = _to_serialized_dataframe(
        df_result[sorted(id_fields)],
        column=COL_DATAPOINT_ID_OBJECT,
    )
    df_result_eval = _to_serialized_dataframe(df_result.drop(columns=id_fields), column=COL_RESULT)
    df_result_eval[COL_EVAL_CONFIG] = json.dumps(eval_config) if eval_config is not None else None
    df_result_eval = pd.concat([df_result_eval, df_serialized_datapoint_id_object], axis=1)
    return df_result_eval


def _send_upload_results_request(
    model: str,
    load_uuid: str,
    dataset_id: int,
    sources: Optional[List[Dict[str, str]]],
) -> UploadResultsResponse:
    request = UploadResultsRequest(
        model=model,
        uuid=load_uuid,
        dataset_id=dataset_id,
        sources=sources,
    )
    response = krequests.post(Path.UPLOAD_RESULTS, json=asdict(request))
    krequests.raise_for_status(response)
    return from_dict(UploadResultsResponse, response.json())


@with_event(EventAPI.Event.FETCH_DATASET_MODEL_RESULT)
def download_results(
    dataset: str,
    model: str,
) -> Tuple[pd.DataFrame, List[Tuple[EvalConfig, pd.DataFrame]]]:
    """
    Download results given dataset name and model name.

    Concat dataset with results:

    ```python
    df_dp, results = download_results("dataset name", "model name")
    for eval_config, df_result in results:
        df_combined = pd.concat([df_dp, df_result], axis=1)
    ```

    :param dataset: The name of the dataset.
    :param model: The name of the model.
    :return: Tuple of DataFrame of datapoints and list of tuples,
             each containing an evaluation configuration and the corresponding DataFrame of results.
    """
    log.info(f"downloading results for model '{model}' on dataset '{dataset}'")
    df = _fetch_results(dataset, model)

    if df.empty:
        df_datapoints = pd.DataFrame()
    else:
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
    log.info(f"downloaded results for model '{model}' on dataset '{dataset}'")
    return df_datapoints, df_results_by_eval


def _validate_configs(configs: List[EvalConfig]) -> None:
    n = len(configs)
    for i in range(n):
        for j in range(i + 1, n):
            if configs[i] == configs[j]:
                raise IncorrectUsageError("duplicate eval configs are invalid")


def _prepare_upload_results_request(
    dataset: str,
    model: str,
    results: Union[DataFrame, List[Tuple[EvalConfig, DataFrame]]],
) -> Tuple[str, int, int]:
    existing_dataset = _load_dataset_metadata(dataset)
    if not existing_dataset:
        raise NotFoundError(f"dataset {dataset} does not exist")

    id_fields = existing_dataset.id_fields

    if isinstance(results, pd.DataFrame) or isinstance(results, Iterator):
        results = [(None, results)]
    load_uuid = init_upload().uuid

    _validate_configs([cfg for cfg, _ in results])
    total_rows = 0
    for config, df_result_input in results:
        log.info(f"uploading test results with configuration {config}" if config else "uploading test results")
        if isinstance(df_result_input, pd.DataFrame):
            total_rows += df_result_input.shape[0]
            validate_dataframe_ids(df_result_input, id_fields)
            validate_dataframe_have_other_columns_besides_ids(df_result_input, id_fields)
            df_results = _process_result(config, df_result_input, id_fields)
            upload_data_frame(df=df_results, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=load_uuid)
        else:
            id_column_validated = False
            for df_result in df_result_input:
                if not id_column_validated:
                    validate_dataframe_ids(df_result, id_fields)
                    validate_dataframe_have_other_columns_besides_ids(df_result, id_fields)
                    id_column_validated = True
                total_rows += df_result.shape[0]
                df_results = _process_result(config, df_result, id_fields)
                upload_data_frame(df=df_results, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=load_uuid)
    dataset_id = existing_dataset.id
    return load_uuid, dataset_id, total_rows


def _upload_results(
    dataset: str,
    model: str,
    results: Union[DataFrame, List[Tuple[EvalConfig, DataFrame]]],
    sources: Optional[List[Dict[str, str]]] = DEFAULT_SOURCES,
) -> UploadResultsResponse:
    load_uuid, dataset_id, total_rows = _prepare_upload_results_request(dataset, model, results)

    response = _send_upload_results_request(model, load_uuid, dataset_id, sources=sources)
    log.info(
        f"uploaded test results for model '{model}' on dataset '{dataset}': "
        f"{total_rows} uploaded, {response.n_inserted} inserted, {response.n_updated} updated",
    )
    return response


@with_event(EventAPI.Event.UPLOAD_DATASET_MODEL_RESULT)
def upload_results(
    dataset: str,
    model: str,
    results: Union[DataFrame, List[Tuple[EvalConfig, DataFrame]]],
) -> None:
    """
    This function is used for uploading the results from a specified model on a given dataset.

    :param dataset: The name of the dataset.
    :param model: The name of the model.
    :param results: Either a DataFrame or a list of tuples, where each tuple consists of
                    an eval configuration and a DataFrame.

    :return: None
    """
    _upload_results(dataset, model, results)
