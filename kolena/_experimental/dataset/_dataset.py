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
from kolena._utils import krequests_v2 as krequests
from kolena._utils.batched_load import _BatchedLoader
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena._utils.consts import BatchSize
from kolena._utils.state import API_V2
from kolena.errors import InputValidationError
from kolena.workflow._datatypes import _deserialize_dataobject
from kolena.workflow._datatypes import _serialize_dataobject
from kolena.workflow._datatypes import DATA_TYPE_FIELD
from kolena.workflow._datatypes import TypedDataObject

COL_DATAPOINT = "datapoint"
TEST_SAMPLE_TYPE = "TEST_SAMPLE"
FIELD_LOCATOR = "locator"
FIELD_TEXT = "text"


class TestSampleType(str, Enum):
    CUSTOM = "TEST_SAMPLE/CUSTOM"
    DOCUMENT = "TEST_SAMPLE/DOCUMENT"
    IMAGE = "TEST_SAMPLE/IMAGE"
    POINT_CLOUD = "TEST_SAMPLE/POINT_CLOUD"
    TEXT = "TEST_SAMPLE/TEXT"
    VIDEO = "TEST_SAMPLE/VIDEO"


_DATAPOINT_TYPE_MAP = {
    "image": TestSampleType.IMAGE.value,
    "application/pdf": TestSampleType.DOCUMENT.value,
    "text": TestSampleType.DOCUMENT.value,
    "video": TestSampleType.VIDEO.value,
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
        return TestSampleType.POINT_CLOUD.value

    return TestSampleType.CUSTOM.value


def _infer_datatype(df: pd.DataFrame) -> Union[pd.DataFrame, str]:
    if FIELD_LOCATOR in df.columns:
        return df[FIELD_LOCATOR].apply(_infer_datatype_value)
    elif FIELD_TEXT in df.columns:
        return TestSampleType.TEXT.value

    return TestSampleType.CUSTOM.value


def _to_serialized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    object_columns = list(df.select_dtypes(include="object").columns)
    result = df.select_dtypes(exclude="object")
    result[object_columns] = df[object_columns].applymap(_serialize_dataobject)
    result[DATA_TYPE_FIELD] = _infer_datatype(df)
    result[COL_DATAPOINT] = result.to_dict("records")
    result[COL_DATAPOINT] = result[COL_DATAPOINT].apply(lambda x: json.dumps(x, sort_keys=True))

    return result[[COL_DATAPOINT]]


def _to_deserialized_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    flattened = pd.json_normalize([json.loads(r[COL_DATAPOINT]) for r in df.to_dict("records")], max_level=1)
    flattened = flattened.loc[:, ~flattened.columns.str.endswith(DATA_TYPE_FIELD)]
    object_columns = list(flattened.select_dtypes(include="object").columns)
    result = flattened.select_dtypes(exclude="object")
    result[object_columns] = flattened[object_columns].applymap(_deserialize_dataobject)

    return result


def register_dataset(name: str, df: pd.DataFrame) -> None:
    """
    Create or update a dataset with datapoints.
    """
    load_uuid = init_upload().uuid

    df_serialized = _to_serialized_dataframe(df)
    upload_data_frame(df=df_serialized, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=load_uuid)
    request = RegisterRequest(name=name, uuid=load_uuid)
    response = krequests.post(Path.REGISTER, json=asdict(request))
    krequests.raise_for_status(response)


def iter_dataset(
    name: str,
    batch_size: int = BatchSize.LOAD_SAMPLES.value,
) -> Iterator[pd.DataFrame]:
    """
    Get an interator over datapoints in the dataset.
    """
    if batch_size <= 0:
        raise InputValidationError(f"invalid batch_size '{batch_size}': expected positive integer")
    init_request = LoadDatapointsRequest(
        name=name,
        batch_size=batch_size,
    )
    for df_batch in _BatchedLoader.iter_data(
        init_request=init_request,
        endpoint_path=Path.LOAD_DATAPOINTS.value,
        df_class=None,
        endpoint_api_version=API_V2,
    ):
        yield _to_deserialized_dataframe(df_batch)


def fetch_dataset(
    name: str,
    batch_size: int = BatchSize.LOAD_SAMPLES.value,
) -> pd.DataFrame:
    df_batches = list(iter_dataset(name, batch_size))
    return pd.concat(df_batches, ignore_index=True) if df_batches else pd.DataFrame()
