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
import mimetypes
import sys
from dataclasses import asdict
from enum import Enum
from typing import Any
from typing import Dict
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union
from urllib.parse import urlparse

import pandas as pd
import requests

from kolena._api.v1.event import EventAPI
from kolena._api.v2.dataset import CommitData
from kolena._api.v2.dataset import EntityData
from kolena._api.v2.dataset import ListCommitHistoryRequest
from kolena._api.v2.dataset import ListCommitHistoryResponse
from kolena._api.v2.dataset import LoadDatapointsRequest
from kolena._api.v2.dataset import LoadDatasetByNameRequest
from kolena._api.v2.dataset import Path
from kolena._api.v2.dataset import RegisterRequest
from kolena._utils import krequests_v2 as krequests
from kolena._utils import log
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.dataframes.transformers import df_apply
from kolena._utils.dataframes.transformers import json_normalize
from kolena._utils.datatypes import _deserialize_dataobject
from kolena._utils.datatypes import _serialize_dataobject
from kolena._utils.datatypes import DATA_TYPE_FIELD
from kolena._utils.endpoints import get_dataset_url
from kolena._utils.instrumentation import with_event
from kolena._utils.serde import from_dict
from kolena._utils.state import API_V2
from kolena.dataset._common import COL_DATAPOINT
from kolena.dataset._common import COL_DATAPOINT_ID_OBJECT
from kolena.dataset._common import DEFAULT_SOURCES
from kolena.dataset._common import validate_batch_size
from kolena.dataset._common import validate_dataframe_ids
from kolena.errors import InputValidationError
from kolena.io import _dataframe_object_serde

_FIELD_ID = "id"
_FIELD_LOCATOR = "locator"
_FIELD_TEXT = "text"
_SEP = "."


class DatapointType(str, Enum):
    AUDIO = "DATAPOINT/AUDIO"
    COMPOSITE = "DATAPOINT/COMPOSITE"
    DOCUMENT = "DATAPOINT/DOCUMENT"
    IMAGE = "DATAPOINT/IMAGE"
    POINT_CLOUD = "DATAPOINT/POINT_CLOUD"
    TABULAR = "DATAPOINT/TABULAR"
    TEXT = "DATAPOINT/TEXT"
    VIDEO = "DATAPOINT/VIDEO"

    @classmethod
    def has_value(cls, item: Any) -> bool:
        return item in cls.__members__.values()


_DATAPOINT_TYPE_MAP = {
    "image": DatapointType.IMAGE.value,
    "application/pdf": DatapointType.DOCUMENT.value,
    "text": DatapointType.DOCUMENT.value,
    "video": DatapointType.VIDEO.value,
    "audio": DatapointType.AUDIO.value,
}


def _get_datapoint_type(mimetype_str: str) -> Optional[str]:
    main_type, _ = mimetype_str.split("/")
    return _DATAPOINT_TYPE_MAP.get(mimetype_str, None) or _DATAPOINT_TYPE_MAP.get(main_type, None)


def _normalize_url(x: str) -> str:
    url = urlparse(x)
    return url._replace(query="", fragment="").geturl()


def _infer_datatype_value(x: str) -> str:
    url = _normalize_url(x)
    mtype, _ = mimetypes.guess_type(url)
    if mtype:
        datatype = _get_datapoint_type(mtype)
        if datatype is not None:
            return datatype
    elif url.endswith(".pcd"):
        return DatapointType.POINT_CLOUD.value

    return DatapointType.TABULAR.value


def _add_datatype(df: pd.DataFrame) -> None:
    """Adds `data_type` column(s) to input DataFrame."""
    prefixes = {
        column.rsplit(sep=_SEP, maxsplit=1)[0]
        for column in df.columns.values
        if isinstance(column, str) and _SEP in column
    }
    if prefixes:
        df[DATA_TYPE_FIELD] = DatapointType.COMPOSITE.value
        for prefix in prefixes:
            if not prefix.strip():
                raise InputValidationError(
                    "Empty prefix encountered when parsing composite dataset. "
                    f"Columns must lead with at least one non-whitespace character prior to delimeter '{_SEP}'.",
                )
            if prefix in df.columns:
                raise InputValidationError(
                    f"Conflicting column '{prefix}' encountered when formatting composite dataset.",
                )
            if _SEP in prefix:
                raise InputValidationError(
                    f"More than one delimeter '{_SEP}' in prefix: '{prefix}'.",
                )

            composite_columns = df.filter(regex=rf"^{prefix}", axis=1).columns.to_list()
            composite = df.loc[:, composite_columns].rename(columns=lambda col: col.split(_SEP)[-1])
            composite[DATA_TYPE_FIELD] = _infer_datatype(composite)

            df[prefix] = composite.to_dict("records")
            df.drop(columns=composite_columns, inplace=True)
    else:
        df[DATA_TYPE_FIELD] = _infer_datatype(df)


def _infer_datatype(df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    if _FIELD_LOCATOR in df.columns:
        return df_apply(df[_FIELD_LOCATOR], _infer_datatype_value)
    elif _FIELD_TEXT in df.columns:
        return DatapointType.TEXT.value

    return DatapointType.TABULAR.value


def _infer_id_fields(df: pd.DataFrame) -> List[str]:
    def get_id_fields_by(field: str) -> List[str]:
        return [
            id_field
            for id_field in df.columns.array
            if isinstance(id_field, str) and id_field.rsplit(_SEP, maxsplit=1)[-1] == field
        ]

    if id_fields := get_id_fields_by(_FIELD_ID):
        return id_fields
    elif id_fields := get_id_fields_by(_FIELD_LOCATOR):
        return id_fields
    elif id_fields := get_id_fields_by(_FIELD_TEXT):
        return id_fields
    raise InputValidationError("Failed to infer the id_fields, please provide id_fields explicitly")


def _to_serialized_dataframe(df: pd.DataFrame, column: str) -> pd.DataFrame:
    result = _dataframe_object_serde(df, _serialize_dataobject)
    if column == COL_DATAPOINT:
        _add_datatype(result)
    result[column] = result.to_dict("records")
    result[column] = df_apply(result[column], lambda x: json.dumps(x))

    return result[[column]]


def _to_deserialized_dataframe(df: pd.DataFrame, column: str) -> pd.DataFrame:
    flattened = json_normalize(
        [json.loads(r[column]) if r[column] is not None else {} for r in df.to_dict("records")],
        max_level=0,
    )
    flattened = _flatten_composite(flattened)
    flattened = flattened.loc[:, ~flattened.columns.str.endswith(DATA_TYPE_FIELD)]
    df_post = _dataframe_object_serde(flattened, _deserialize_dataobject)
    return df_post


def _flatten_composite(df: pd.DataFrame) -> pd.DataFrame:
    for key, value in df.iloc[0].items():
        if isinstance(value, dict) and DatapointType.has_value(value.get(DATA_TYPE_FIELD)):
            flattened = json_normalize(df[key], max_level=0).rename(
                columns=lambda col: f"{key}{_SEP}{col}",
            )
            df = df.join(flattened)
            df.drop(columns=[key], inplace=True)
    return df


def _upload_dataset_chunk(df: pd.DataFrame, load_uuid: str, id_fields: List[str]) -> None:
    df_serialized_datapoint = _to_serialized_dataframe(df, column=COL_DATAPOINT)
    df_serialized_datapoint_id_object = _to_serialized_dataframe(df[sorted(id_fields)], column=COL_DATAPOINT_ID_OBJECT)
    df_serialized = pd.concat([df_serialized_datapoint, df_serialized_datapoint_id_object], axis=1)

    upload_data_frame(df=df_serialized, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=load_uuid)


def _load_dataset_metadata(name: str) -> Optional[EntityData]:
    """
    Load the metadata of a given dataset.

    :param name: The name of the dataset.
    :return: The metadata of the dataset.
    """
    response = krequests.put(Path.LOAD_DATASET, json=asdict(LoadDatasetByNameRequest(name=name)))
    if response.status_code == requests.codes.not_found:
        return None

    response.raise_for_status()

    return from_dict(EntityData, response.json())


def _resolve_id_fields(
    df: pd.DataFrame,
    id_fields: Optional[List[str]],
    existing_dataset: Optional[EntityData],
) -> List[str]:
    existing_id_fields = []
    if existing_dataset:
        existing_id_fields = existing_dataset.id_fields
    if not id_fields:
        if existing_id_fields:
            return existing_id_fields
        else:
            id_fields = _infer_id_fields(df)
    return id_fields


def _prepare_upload_dataset_request(
    name: str,
    df: Union[pd.DataFrame, Iterator[pd.DataFrame]],
    *,
    id_fields: Optional[List[str]] = None,
) -> Tuple[List[str], str]:
    load_uuid = init_upload().uuid
    existing_dataset = _load_dataset_metadata(name)
    if isinstance(df, pd.DataFrame):
        id_fields = _resolve_id_fields(df, id_fields, existing_dataset)
        validate_dataframe_ids(df, id_fields)
        _upload_dataset_chunk(df, load_uuid, id_fields)
    else:
        validated = False
        for chunk in df:
            if not validated:
                id_fields = _resolve_id_fields(chunk, id_fields, existing_dataset)
                validate_dataframe_ids(chunk, id_fields)
                validated = True
            assert id_fields is not None
            _upload_dataset_chunk(chunk, load_uuid, id_fields)
    assert id_fields is not None
    return id_fields, load_uuid


def _send_upload_dataset_request(
    name: str,
    id_fields: List[str],
    load_uuid: str,
    sources: Optional[List[Dict[str, str]]],
) -> EntityData:
    request = RegisterRequest(name=name, id_fields=id_fields, uuid=load_uuid, sources=sources)
    response = krequests.post(Path.REGISTER, json=asdict(request))
    krequests.raise_for_status(response)
    data = from_dict(EntityData, response.json())
    return data


def _upload_dataset(
    name: str,
    df: Union[pd.DataFrame, Iterator[pd.DataFrame]],
    *,
    id_fields: Optional[List[str]] = None,
    sources: Optional[List[Dict[str, str]]] = DEFAULT_SOURCES,
) -> None:
    prepared_id_fields, load_uuid = _prepare_upload_dataset_request(name, df, id_fields=id_fields)

    data = _send_upload_dataset_request(name, prepared_id_fields, load_uuid, sources=sources)
    log.info(f"uploaded dataset '{name}' ({get_dataset_url(dataset_id=data.id)})")


@with_event(event_name=EventAPI.Event.REGISTER_DATASET)
def upload_dataset(
    name: str,
    df: Union[pd.DataFrame, Iterator[pd.DataFrame]],
    *,
    id_fields: Optional[List[str]] = None,
) -> None:
    """
    Create or update a dataset with the contents of the provided DataFrame `df`.

    !!! note "Updating `id_fields`"
        ID fields are used to associate model results (uploaded via [`upload_results`][kolena.dataset.upload_results])
        with datapoints in this dataset. When updating an existing dataset, update `id_fields` with caution.

    :param name: The name of the dataset.
    :param df: A DataFrame or iterator of DataFrames. Provide an iterator to perform batch upload (example:
        `csv_reader = pd.read_csv("PathToDataset.csv", chunksize=10)`).
    :param id_fields: Optionally specify a list of ID fields that will be used to link model results with the datapoints
        within a dataset. When unspecified, a suitable value is inferred from the columns of the provided `df`. Note
        that `id_fields` must be hashable.
    """
    _upload_dataset(name, df, id_fields=id_fields)


def _iter_dataset_raw(
    name: str,
    commit: Optional[str] = None,
    batch_size: int = BatchSize.LOAD_SAMPLES.value,
) -> Iterator[pd.DataFrame]:
    validate_batch_size(batch_size)
    init_request = LoadDatapointsRequest(
        name=name,
        commit=commit,
        batch_size=batch_size,
    )
    yield from _BatchedLoader.iter_data(
        init_request=init_request,
        endpoint_path=Path.LOAD_DATAPOINTS.value,
        df_class=None,
        endpoint_api_version=API_V2,
    )


def _iter_dataset(
    name: str,
    commit: Optional[str] = None,
    batch_size: int = BatchSize.LOAD_SAMPLES.value,
) -> Iterator[pd.DataFrame]:
    """
    Get an iterator over datapoints in the dataset.
    """
    for df_batch in _iter_dataset_raw(name, commit, batch_size):
        yield _to_deserialized_dataframe(df_batch, column=COL_DATAPOINT)


@with_event(event_name=EventAPI.Event.FETCH_DATASET)
def download_dataset(name: str, *, commit: Optional[str] = None) -> pd.DataFrame:
    """
    Download an entire dataset given its name.

    :param name: The name of the dataset.
    :param commit: The commit hash for version control. Get the latest commit when this value is `None`.
    :return: A DataFrame containing the specified dataset.
    """
    df_batches = list(_iter_dataset(name, commit, BatchSize.LOAD_SAMPLES.value))
    log.info(f"downloaded dataset '{name}'")
    df_dataset = pd.concat(df_batches, ignore_index=True) if df_batches else pd.DataFrame()
    return df_dataset


def _list_commits(name: str, descending: bool = False, offset: int = 0, limit: int = 50) -> ListCommitHistoryResponse:
    """
    Invoke the list-commits api.
    """
    request = ListCommitHistoryRequest(name=name, descending=descending, offset=offset, limit=limit)
    response = krequests.put(Path.LIST_COMMITS, json=asdict(request))
    krequests.raise_for_status(response)
    return from_dict(ListCommitHistoryResponse, response.json())


def _iter_commits(
    name: str,
    descending: bool = False,
    limit: Optional[int] = None,
    page_size: int = 50,
) -> Iterator[Tuple[int, List[CommitData]]]:
    """
    Get an iterator over the commit history of the dataset.
    """
    current_count = 0
    if limit is None:
        limit = sys.maxsize
    while True:
        response = _list_commits(name, descending=descending, offset=current_count, limit=page_size)
        yield response.total_count, response.records
        current_count += len(response.records)
        if current_count >= min(limit, response.total_count):
            break


@with_event(event_name=EventAPI.Event.FETCH_DATASET_HISTORY)
def _fetch_dataset_history(
    name: str,
    *,
    descending: bool = False,
    limit: Optional[int] = None,
    page_size: int = 50,
) -> Tuple[int, List[CommitData]]:
    """
    Get the commit history of a dataset.

    :param name: The name of the dataset
    :param descending: If True, return the results in descending order.
    :param limit: The maximum number of results to return. If None, all results will be returned.
    :param page_size: The number of items to return per page.
    :return: A tuple where the first element is the total number of commits
             and the second element is a list of CommitData objects.
    """
    iter_commit_responses = list(_iter_commits(name, descending, limit, page_size))
    total_commit_count = iter_commit_responses[0][0]
    commits = [commit for response in iter_commit_responses for commit in response[1]][:limit]
    return total_commit_count, commits
