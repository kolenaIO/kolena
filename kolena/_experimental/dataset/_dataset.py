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
from typing import Union

import pandas as pd

from kolena._api.v2.dataset import LoadDatapointsRequest
from kolena._api.v2.dataset import Path
from kolena._api.v2.dataset import RegisterRequest
from kolena._experimental.dataset.common import COL_DATAPOINT
from kolena._experimental.dataset.common import validate_batch_size
from kolena._utils import krequests_v2 as krequests
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.state import API_V2
from kolena.workflow._datatypes import _deserialize_dataobject
from kolena.workflow._datatypes import _serialize_dataobject
from kolena.workflow._datatypes import DATA_TYPE_FIELD
from kolena.workflow._datatypes import TypedDataObject
from kolena.workflow.io import _dataframe_object_serde


FIELD_LOCATOR = "locator"
FIELD_TEXT = "text"


class DatapointType(str, Enum):
    AUDIO = "DATAPOINT/AUDIO"
    DOCUMENT = "DATAPOINT/DOCUMENT"
    IMAGE = "DATAPOINT/IMAGE"
    POINT_CLOUD = "DATAPOINT/POINT_CLOUD"
    TABULAR = "DATAPOINT/TABULAR"
    TEXT = "DATAPOINT/TEXT"
    VIDEO = "DATAPOINT/VIDEO"


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


def _infer_datatype(df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    if FIELD_LOCATOR in df.columns:
        return df[FIELD_LOCATOR].apply(_infer_datatype_value)
    elif FIELD_TEXT in df.columns:
        return DatapointType.TEXT.value

    return DatapointType.TABULAR.value


def _to_serialized_dataframe(df: pd.DataFrame, column: str) -> pd.DataFrame:
    result = _dataframe_object_serde(df, _serialize_dataobject)
    if column == COL_DATAPOINT:
        result[DATA_TYPE_FIELD] = _infer_datatype(df)
    result[column] = result.to_dict("records")
    result[column] = result[column].apply(lambda x: json.dumps(x))

    return result[[column]]


def _to_deserialized_dataframe(df: pd.DataFrame, column: str) -> pd.DataFrame:
    flattened = pd.json_normalize(
        [json.loads(r[column]) if r[column] is not None else {} for r in df.to_dict("records")],
        max_level=0,
    )
    flattened = flattened.loc[:, ~flattened.columns.str.endswith(DATA_TYPE_FIELD)]
    result = _dataframe_object_serde(flattened, _deserialize_dataobject)

    return result


def register_dataset(name: str, df: pd.DataFrame) -> None:
    """
    Create or update a dataset with datapoints.
    """
    load_uuid = init_upload().uuid

    df_serialized = _to_serialized_dataframe(df, column=COL_DATAPOINT)
    upload_data_frame(df=df_serialized, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=load_uuid)
    request = RegisterRequest(name=name, uuid=load_uuid)
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
