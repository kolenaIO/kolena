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
import mimetypes
from dataclasses import asdict
from enum import Enum
from typing import Iterator
from typing import List
from typing import Optional
from typing import Union

import pandas as pd
import requests

from kolena._api.v2.dataset import EntityData
from kolena._api.v2.dataset import LoadDatapointsRequest
from kolena._api.v2.dataset import LoadDatasetByNameRequest
from kolena._api.v2.dataset import Path
from kolena._api.v2.dataset import RegisterRequest
from kolena._experimental.dataset.common import COL_DATAPOINT
from kolena._experimental.dataset.common import COL_DATAPOINT_ID_OBJECT
from kolena._experimental.dataset.common import validate_batch_size
from kolena._experimental.dataset.common import validate_dataframe_ids
from kolena._utils import krequests_v2 as krequests
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.serde import from_dict
from kolena._utils.state import API_V2
from kolena.errors import InputValidationError
from kolena.workflow._datatypes import _deserialize_dataobject
from kolena.workflow._datatypes import _serialize_dataobject
from kolena.workflow._datatypes import DATA_TYPE_FIELD
from kolena.workflow._datatypes import TypedDataObject
from kolena.workflow.io import _dataframe_object_serde

FIELD_LOCATOR = "locator"
FIELD_TEXT = "text"
SEP = "."


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
    def has_value(cls, item) -> bool:
        return item in cls.__members__.values()


_DATAPOINT_TYPE_MAP = {
    "image": DatapointType.IMAGE.value,
    "application/pdf": DatapointType.DOCUMENT.value,
    "text": DatapointType.DOCUMENT.value,
    "video": DatapointType.VIDEO.value,
    "audio": DatapointType.AUDIO.value,
}


def _dataobject_type(obj: TypedDataObject) -> str:
    obj_type = obj._data_type()
    return f"{obj_type._data_category()}/{obj_type.value}"


def _get_datapoint_type(mimetype_str: str) -> str:
    main_type, sub_type = mimetype_str.split("/")
    return _DATAPOINT_TYPE_MAP.get(mimetype_str, None) or _DATAPOINT_TYPE_MAP.get(main_type, None)


def _infer_datatype_value(x: str) -> str:
    mtype, _ = mimetypes.guess_type(x)
    if mtype:
        datatype = _get_datapoint_type(mtype)
        if datatype is not None:
            return datatype
    elif x.endswith(".pcd"):
        return DatapointType.POINT_CLOUD.value

    return DatapointType.TABULAR.value


def _add_datatype(df: pd.DataFrame) -> None:
    """Adds `data_type` column(s) to input DataFrame."""
    prefixes = {
        column.rsplit(sep=SEP, maxsplit=1)[0]
        for column in df.columns.values
        if isinstance(column, str) and SEP in column
    }
    if prefixes:
        df[DATA_TYPE_FIELD] = DatapointType.COMPOSITE.value
        for prefix in prefixes:
            if not prefix.strip():
                raise InputValidationError(
                    "Empty prefix encountered when parsing composite dataset. "
                    f"Columns must lead with at least one non-whitespace character prior to delimeter '{SEP}'.",
                )
            if prefix in df.columns:
                raise InputValidationError(
                    f"Conflicting column '{prefix}' encountered when formatting composite dataset.",
                )
            if SEP in prefix:
                raise InputValidationError(
                    f"More than one delimeter '{SEP}' in prefix: '{prefix}'.",
                )

            composite_columns = df.filter(regex=rf"^{prefix}", axis=1).columns.to_list()
            composite = df.loc[:, composite_columns].rename(columns=lambda col: col.split(SEP)[-1])
            composite[DATA_TYPE_FIELD] = _infer_datatype(composite)

            df[prefix] = composite.to_dict("records")
            df.drop(columns=composite_columns, inplace=True)
    else:
        df[DATA_TYPE_FIELD] = _infer_datatype(df)


def _infer_datatype(df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    if FIELD_LOCATOR in df.columns:
        return df[FIELD_LOCATOR].apply(_infer_datatype_value)
    elif FIELD_TEXT in df.columns:
        return DatapointType.TEXT.value

    return DatapointType.TABULAR.value


def _infer_id_fields(df: pd.DataFrame) -> List[str]:
    def get_id_fields_by(field: str) -> List[str]:
        return [
            id_field
            for id_field in df.columns.array
            if isinstance(id_field, str) and id_field.rsplit(SEP, maxsplit=1)[-1] == field
        ]

    if id_fields := get_id_fields_by(FIELD_LOCATOR):
        return id_fields
    elif id_fields := get_id_fields_by(FIELD_TEXT):
        return id_fields
    raise InputValidationError("Failed to infer the id_fields, please provide id_fields explicitly")


def _to_serialized_dataframe(df: pd.DataFrame, column: str) -> pd.DataFrame:
    result = _dataframe_object_serde(df, _serialize_dataobject)
    if column == COL_DATAPOINT:
        _add_datatype(result)
    result[column] = result.to_dict("records")
    result[column] = result[column].apply(lambda x: json.dumps(x))

    return result[[column]]


def _to_deserialized_dataframe(df: pd.DataFrame, column: str) -> pd.DataFrame:
    flattened = pd.json_normalize(
        [json.loads(r[column]) if r[column] is not None else {} for r in df.to_dict("records")],
        max_level=0,
    )
    flattened = _flatten_composite(flattened)
    flattened = flattened.loc[:, ~flattened.columns.str.endswith(DATA_TYPE_FIELD)]
    return _dataframe_object_serde(flattened, _deserialize_dataobject)


def _flatten_composite(df: pd.DataFrame) -> pd.DataFrame:
    for key, value in df.iloc[0].items():
        if isinstance(value, dict) and DatapointType.has_value(value.get(DATA_TYPE_FIELD)):
            flattened = pd.json_normalize(df[key], max_level=0).rename(
                columns=lambda col: f"{key}{SEP}{col}",
            )
            df = df.join(flattened)
            df.drop(columns=[key], inplace=True)
    return df


def _upload_dataset_chunk(df: pd.DataFrame, load_uuid: str, id_fields: List[str]) -> None:
    df_serialized_datapoint = _to_serialized_dataframe(df, column=COL_DATAPOINT)
    df_serialized_datapoint_id_object = _to_serialized_dataframe(df[sorted(id_fields)], column=COL_DATAPOINT_ID_OBJECT)
    df_serialized = pd.concat([df_serialized_datapoint, df_serialized_datapoint_id_object], axis=1)

    upload_data_frame(df=df_serialized, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=load_uuid)


def load_dataset(name: str) -> Optional[EntityData]:
    response = krequests.put(Path.LOAD_DATASET, json=asdict(LoadDatasetByNameRequest(name=name)))
    if response.status_code == requests.codes.not_found:
        return None

    response.raise_for_status()

    return from_dict(EntityData, response.json())


def resolve_id_fields(
    df: pd.DataFrame,
    id_fields: Optional[List[str]],
    existing_dataset: Optional[EntityData],
) -> List[str]:
    existing_id_fields = []
    if existing_dataset:
        existing_id_fields = existing_dataset.id_fields
    if not id_fields:
        if existing_id_fields:
            raise InputValidationError("id_fields is required for updating an existing dataset")
        else:
            id_fields = _infer_id_fields(df)
    return id_fields


def register_dataset(
    name: str,
    df: Union[Iterator[pd.DataFrame], pd.DataFrame],
    id_fields: Optional[List[str]] = None,
) -> None:
    """
    Create or update a dataset with datapoints and id_fields. If the dataset already exists, in order to associate the
    existing result with the new datapoints, the id_fields need be the same as the existing dataset.

    :param name: name of the dataset
    :param df: an iterator of pandas dataframe or a pandas dataframe, you can pass in the iterator if you want to have
                batch processing,
                 example iterator usage: csv_reader = pd.read_csv("PathToDataset.csv", chunksize=10)
    :param id_fields: a list of id fields, this will be used to link the result with the datapoints, if this is not
                 provided, it will be inferred from the dataset
    :return None
    """
    load_uuid = init_upload().uuid
    existing_dataset = load_dataset(name)
    if isinstance(df, pd.DataFrame):
        id_fields = resolve_id_fields(df, id_fields, existing_dataset)
        validate_dataframe_ids(df, id_fields)
        _upload_dataset_chunk(df, load_uuid, id_fields)
    else:
        validated = False
        for chunk in df:
            if not validated:
                id_fields = resolve_id_fields(chunk, id_fields, existing_dataset)
                validate_dataframe_ids(chunk, id_fields)
                validated = True
            _upload_dataset_chunk(chunk, load_uuid, id_fields)
    request = RegisterRequest(name=name, id_fields=id_fields, uuid=load_uuid)
    response = krequests.post(Path.REGISTER, json=asdict(request))
    krequests.raise_for_status(response)


def _iter_dataset_raw(
    name: str,
    batch_size: int = BatchSize.LOAD_SAMPLES.value,
) -> Iterator[pd.DataFrame]:
    validate_batch_size(batch_size)
    init_request = LoadDatapointsRequest(
        name=name,
        batch_size=batch_size,
    )
    yield from _BatchedLoader.iter_data(
        init_request=init_request,
        endpoint_path=Path.LOAD_DATAPOINTS.value,
        df_class=None,
        endpoint_api_version=API_V2,
    )


def iter_dataset(
    name: str,
    batch_size: int = BatchSize.LOAD_SAMPLES.value,
) -> Iterator[pd.DataFrame]:
    """
    Get an iterator over datapoints in the dataset.
    """
    for df_batch in _iter_dataset_raw(name, batch_size):
        yield _to_deserialized_dataframe(df_batch, column=COL_DATAPOINT)


def fetch_dataset(
    name: str,
    batch_size: int = BatchSize.LOAD_SAMPLES.value,
) -> pd.DataFrame:
    """
    Fetch an entire dataset given its name.
    """
    df_batches = list(iter_dataset(name, batch_size))
    return pd.concat(df_batches, ignore_index=True) if df_batches else pd.DataFrame()
