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
from kolena._api.v2.model import Path
from kolena._api.v2.model import UploadInferencesRequest
from kolena._experimental.dataset._dataset import _iter_dataset_raw
from kolena._experimental.dataset._dataset import _to_deserialized_dataframe
from kolena._experimental.dataset._dataset import _to_serialized_dataframe
from kolena._experimental.dataset.common import COL_DATAPOINT
from kolena._experimental.dataset.common import COL_EVAL_CONFIG
from kolena._experimental.dataset.common import COL_INFERENCE
from kolena._experimental.dataset.common import COL_METRICS
from kolena._experimental.dataset.common import validate_batch_size
from kolena._utils import krequests_v2 as krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.state import API_V2
from kolena.errors import IncorrectUsageError


TYPE_EVALUATION_CONFIG = Dict[str, Any]
INFER_FUNC_TYPE = Callable[[pd.DataFrame], pd.DataFrame]
EVAL_FUNC_TYPE = Callable[[pd.DataFrame, pd.DataFrame, Optional[TYPE_EVALUATION_CONFIG]], pd.DataFrame]


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


def test(
    dataset: str,
    model: str,
    infer: Optional[INFER_FUNC_TYPE] = None,
    eval: Optional[EVAL_FUNC_TYPE] = None,
    eval_configs: Optional[Union[TYPE_EVALUATION_CONFIG, List[TYPE_EVALUATION_CONFIG]]] = None,
) -> None:
    """
    This function is used for running inference and evaluation on a given dataset using a specified model.

    :param dataset: The name of the dataset to be used.
    :param model: The name of the model to be used.
    :param infer: The inference function to be used. Defaults to None.
    :param eval: The evaluation function to be used if any. Defaults to None.
    :param eval_configs: The evaluation configurations to be used. Defaults to None.

    :return None: This function doesn't return anything.
    """
    if infer:
        df_data = _fetch_dataset(dataset)
        df_datapoints = _to_deserialized_dataframe(df_data, column=COL_DATAPOINT)
        log.info(f"fetched {len(df_data)} for dataset {dataset}")

        df_inferences = infer(df_datapoints)
        validate_data(df_datapoints, df_inferences)

        df_data["inference"] = _to_serialized_dataframe(df_inferences, column=COL_INFERENCE)
        _upload_inferences(model, df_data)
        log.info(f"uploaded {len(df_inferences)} inferences")

    if eval:
        if not isinstance(eval_configs, list):
            eval_configs = [eval_configs]

        df_data = _fetch_inferences(dataset, model)
        log.info(f"fetched {len(df_data)} inferences")
        df_datapoints = _to_deserialized_dataframe(df_data, column=COL_DATAPOINT)
        df_inferences = _to_deserialized_dataframe(df_data, column=COL_INFERENCE)

        metrics = []
        for config in eval_configs:
            log.info(f"start evaluation with configuration {config}" if config else "start evaluation")
            single_metrics = eval(df_datapoints, df_inferences, config)
            validate_data(df_data, single_metrics)
            metrics.append(single_metrics)
            log.info(f"completed evaluation with configuration {config}" if config else "completed evaluation")

        df_metrics = _process_metrics(df_data, list(zip(eval_configs, metrics)))
        n_uploaded_metrics = _upload_metrics(df_metrics)
        log.info(f"uploaded {n_uploaded_metrics} evaluation results")
