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
from typing import Any
from typing import Callable
from typing import Union

import pandas as pd

from kolena.workflow import DataObject
from kolena.workflow._datatypes import _DATA_TYPE_MAP
from kolena.workflow._datatypes import DATA_TYPE_FIELD


def _serialize_dataobject(x: Any) -> Any:
    if isinstance(x, list):
        return [item._to_dict() if isinstance(item, DataObject) else item for item in x]

    return x._to_dict() if isinstance(x, DataObject) else x


def _deserialize_dataobject(x: Any) -> Any:
    if isinstance(x, list):
        return [_deserialize_dataobject(item) for item in x]

    if isinstance(x, dict) and DATA_TYPE_FIELD in x:
        data = {**x}
        data_type = data.pop(DATA_TYPE_FIELD)
        typed_dataobject = _DATA_TYPE_MAP.get(data_type, None)
        if typed_dataobject:
            return typed_dataobject._from_dict(data)

    return x


def _serialize_dataobject_str(x: Any) -> Any:
    y = _serialize_dataobject(x)
    if isinstance(y, list) or isinstance(y, dict):
        return json.dumps(y)

    return y


def _deserialize_dataobject_str(x: Any) -> Any:
    y = x
    if isinstance(x, str):
        try:
            y = json.loads(x)
        except Exception:
            ...

    return _deserialize_dataobject(y)


def dataframe_to_csv(df: pd.DataFrame, *args, **kwargs) -> Union[str, None]:
    """
    Helper function to export pandas DataFrame containing annotation or asset to CSV format.

    :param args: positional arguments to `pandas.DataFrame.to_csv`.
    :param kwargs: keyword arguments to `pandas.DataFrame.to_csv`.
    :return: None or str.
    """
    df_post = _dataframe_object_serde(df, _serialize_dataobject_str)
    return df_post.to_csv(*args, **kwargs)


def dataframe_from_csv(*args, **kwargs) -> pd.DataFrame:
    """
    Helper function to load pandas DataFrame exported to CSV with `dataframe_to_csv`.

    :param args: positional arguments to `pandas.DataFrame.read_csv`.
    :param kwargs: keyword arguments to `pandas.DataFrame.read_csv`.
    :return: DataFrame.
    """
    df = pd.read_csv(*args, **kwargs)
    df_post = _dataframe_object_serde(df, _deserialize_dataobject_str)

    return df_post


def dataframe_from_json(*args, **kwargs) -> pd.DataFrame:
    """
    Helper function to load pandas DataFrame containing annotation or asset from JSON file or string.

    :param args: positional arguments to `pandas.DataFrame.read_json`.
    :param kwargs: keyword arguments to `pandas.DataFrame.read_json`.
    :return: DataFrame.
    """
    df = pd.read_json(*args, **kwargs)
    df_post = _dataframe_object_serde(df, _deserialize_dataobject)

    return df_post


def _dataframe_object_serde(df, serde_fn: Callable[[Any], Any]):
    columns = list(df.columns)
    df_post = pd.DataFrame(columns=columns)
    for column in columns:
        if df.dtypes[column] == "object":
            df_post[column] = df[column].apply(serde_fn)
        else:
            df_post[column] = df[column]
    return df_post
