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

from kolena._api.v2.model import LoadResultsRequest
from kolena._api.v2.model import Path
from kolena._api.v2.model import UploadResultsRequest
from kolena._experimental.dataset._dataset import _iter_dataset_raw
from kolena._experimental.dataset._dataset import _to_deserialized_dataframe
from kolena._experimental.dataset._dataset import _to_serialized_dataframe
from kolena._experimental.dataset._dataset import load_dataset
from kolena._experimental.dataset.common import COL_DATAPOINT
from kolena._experimental.dataset.common import COL_DATAPOINT_ID_OBJECT
from kolena._experimental.dataset.common import COL_EVAL_CONFIG
from kolena._experimental.dataset.common import COL_RESULT
from kolena._experimental.dataset.common import validate_batch_size
from kolena._experimental.dataset.common import validate_dataframe_ids
from kolena._utils import krequests_v2 as krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.state import API_V2
from kolena.errors import IncorrectUsageError
from kolena.errors import NotFoundError

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


def _process_result(
    eval_config: Optional[TYPE_EVALUATION_CONFIG],
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


def _upload_results(
    model: str,
    load_uuid: str,
    dataset_id: int,
) -> None:
    request = UploadResultsRequest(
        model=model,
        uuid=load_uuid,
        dataset_id=dataset_id,
    )
    response = krequests.post(Path.UPLOAD_RESULTS, json=asdict(request))
    krequests.raise_for_status(response)


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


def test(
    dataset: str,
    model: str,
    results: Union[
        pd.DataFrame,
        Iterator[pd.DataFrame],
        List[
            Tuple[
                TYPE_EVALUATION_CONFIG,
                Union[
                    pd.DataFrame,
                    Iterator[pd.DataFrame],
                ],
            ]
        ],
    ],
) -> None:
    """
    This function is used for testing a specified model on a given dataset.

    :param dataset: The name of the dataset to be used.
    :param model: The name of the model to be used.
    :param results: Either a DataFrame or a list of tuples, where each tuple consists of
                    a eval configuration and a DataFrame.
    :return None
    """
    existing_dataset = load_dataset(dataset)
    if not existing_dataset:
        raise NotFoundError(f"dataset {dataset} does not exist")

    if isinstance(results, pd.DataFrame) or isinstance(results, Iterator):
        results = [(None, results)]
    load_uuid = init_upload().uuid

    _validate_configs([cfg for cfg, _ in results])
    for config, df_result_input in results:
        log.info(f"start evaluation with configuration {config}" if config else "start evaluation")
        if isinstance(df_result_input, pd.DataFrame):
            validate_dataframe_ids(df_result_input, existing_dataset.id_fields)
            df_results = _process_result(config, df_result_input, existing_dataset.id_fields)
            upload_data_frame(df=df_results, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=load_uuid)
        else:
            id_column_validated = False
            for df_result in df_result_input:
                if not id_column_validated:
                    validate_dataframe_ids(df_result, existing_dataset.id_fields)
                    id_column_validated = True
                df_results = _process_result(config, df_result, existing_dataset.id_fields)
                upload_data_frame(df=df_results, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=load_uuid)

    _upload_results(model, load_uuid, existing_dataset.id)
    log.info(f"uploaded test results for model {model} on dataset {dataset}")
