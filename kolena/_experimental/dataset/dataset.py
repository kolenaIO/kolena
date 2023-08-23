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
from typing import Iterator
from typing import Optional

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
from kolena.workflow._datatypes import _deserialize_series
from kolena.workflow._datatypes import _infer_datatype
from kolena.workflow._datatypes import _serialize_series
from kolena.workflow._datatypes import DATA_TYPE_FIELD
from kolena.workflow._datatypes import TEST_SAMPLE_TYPE


def _to_serialized_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    object_columns = list(df.select_dtypes(include="object").columns)
    result = df.select_dtypes(exclude="object")
    result[object_columns] = df[object_columns].apply(_serialize_series)

    if "locator" in df.columns:
        result[DATA_TYPE_FIELD] = df["locator"].apply(_infer_datatype)
    elif "text" in df.columns:
        result[DATA_TYPE_FIELD] = f"{TEST_SAMPLE_TYPE}/TEXT"

    result["datapoint"] = result.to_dict("records")
    result["datapoint"] = result["datapoint"].apply(json.dumps)

    return result[["datapoint"]]


def _to_deserialized_data_frame(df: pd.DataFrame) -> pd.DataFrame:
    flattened = pd.json_normalize([json.loads(r["datapoint"]) for r in df.to_dict("records")], max_level=1)
    flattened = flattened.loc[:, ~flattened.columns.str.endswith(DATA_TYPE_FIELD)]
    object_columns = list(flattened.select_dtypes(include="object").columns)
    result = flattened.select_dtypes(exclude="object")
    result[object_columns] = flattened[object_columns].apply(_deserialize_series)

    return result


def register_dataset(name: str, df: pd.DataFrame) -> None:
    """
    Create or update a dataset with datapoints.
    """
    load_uuid = init_upload().uuid

    df_serialized = _to_serialized_data_frame(df)
    upload_data_frame(df=df_serialized, batch_size=BatchSize.UPLOAD_RECORDS.value, load_uuid=load_uuid)
    request = RegisterRequest(name=name, uuid=load_uuid)
    response = krequests.post(Path.REGISTER, json=asdict(request))
    krequests.raise_for_status(response)


def fetch_dataset(
    name: str,
    version: Optional[int] = None,
    batch_size: int = BatchSize.LOAD_SAMPLES.value,
) -> Iterator[pd.DataFrame]:
    """
    Get an interator over datapoints in the dataset.
    """
    if batch_size <= 0:
        raise InputValidationError(f"invalid batch_size '{batch_size}': expected positive integer")
    init_request = LoadDatapointsRequest(
        name=name,
        version=version,
        batch_size=batch_size,
    )
    for df_batch in _BatchedLoader.iter_data(
        init_request=init_request,
        endpoint_path=Path.LOAD_DATAPOINTS.value,
        df_class=None,
        endpoint_api_version=API_V2,
    ):
        yield _to_deserialized_data_frame(df_batch)
