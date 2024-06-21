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
"""
`kolena.io` provides helper functions for converting `pandas.DataFrame`s containing
to and from common serializable formats while adhering to the JSON specifications, and preserving
non-primitive data objects.
"""
import json
from functools import partial
from typing import Any
from typing import Callable
from typing import Dict
from typing import Union

import pandas as pd

from kolena._utils.dataframes.transformers import _try_parse
from kolena._utils.dataframes.transformers import df_apply
from kolena._utils.datatypes import _get_data_type
from kolena._utils.datatypes import DATA_TYPE_FIELD
from kolena._utils.datatypes import DataObject


class DataObjectJSONEncoder(json.JSONEncoder):
    def default(self, o: Any) -> Dict:
        if isinstance(o, DataObject):
            return o._to_dict()
        return super().default(o)


def _deserialize_dataobject(x: Any) -> Any:
    if isinstance(x, list):
        return [_deserialize_dataobject(item) for item in x]

    if isinstance(x, dict):
        if data_type := x.pop(DATA_TYPE_FIELD, None):
            if typed_dataobject := _get_data_type(data_type):
                return typed_dataobject._from_dict(x)
        else:
            return {k: _deserialize_dataobject(v) for k, v in x.items()}

    return x


def _serialize_dataobject_str(x: Any) -> Any:
    if isinstance(x, (list, dict, DataObject)):
        return json.dumps(x, cls=DataObjectJSONEncoder)
    return x


def _deserialize_dataobject_str(x: Any) -> Any:
    y = _try_parse(x)
    return _deserialize_dataobject(y)


def dataframe_to_csv(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Union[str, None]:
    """
    Helper function to export pandas DataFrame containing annotation or asset to CSV format.

    :param args: positional arguments to `pandas.DataFrame.to_csv`.
    :param kwargs: keyword arguments to `pandas.DataFrame.to_csv`.
    :return: None or str.
    """
    df_post = _dataframe_object_serde(df, _serialize_dataobject_str)
    return df_post.to_csv(*args, **kwargs)


def dataframe_from_csv(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Helper function to load pandas DataFrame exported to CSV with `dataframe_to_csv`.

    :param args: positional arguments to `pandas.DataFrame.read_csv`.
    :param kwargs: keyword arguments to `pandas.DataFrame.read_csv`.
    :return: DataFrame.
    """
    df = pd.read_csv(*args, **kwargs)
    df_post = _dataframe_object_serde(df, _deserialize_dataobject_str)

    return df_post


def dataframe_to_parquet(df: pd.DataFrame, *args: Any, **kwargs: Any) -> Union[bytes, None]:
    """
    Helper function to export pandas DataFrame containing annotation or asset to Parquet format.

    :param args: positional arguments to `pandas.DataFrame.to_parquet`.
    :param kwargs: keyword arguments to `pandas.DataFrame.to_parquet`.
    :return: None or str.
    """
    df_post = _dataframe_object_serde(df, partial(json.dumps, cls=DataObjectJSONEncoder))
    return df_post.to_parquet(*args, **kwargs)


def dataframe_from_parquet(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Helper function to load pandas DataFrame exported to Parquet with `dataframe_to_parquet`.

    :param args: positional arguments to `pandas.DataFrame.read_parquet`.
    :param kwargs: keyword arguments to `pandas.DataFrame.read_parquet`.
    :return: DataFrame.
    """
    df = pd.read_parquet(*args, **kwargs)
    df_post = _dataframe_object_serde(df, _deserialize_dataobject_str)

    return df_post


def dataframe_from_json(*args: Any, **kwargs: Any) -> pd.DataFrame:
    """
    Helper function to load pandas DataFrame containing annotation or asset from JSON file or string.

    :param args: positional arguments to `pandas.DataFrame.read_json`.
    :param kwargs: keyword arguments to `pandas.DataFrame.read_json`.
    :return: DataFrame.
    """
    df = pd.read_json(*args, **kwargs)
    df_post = _dataframe_object_serde(df, _deserialize_dataobject)

    return df_post


def _dataframe_object_serde(df: pd.DataFrame, serde_fn: Callable[[Any], Any]) -> pd.DataFrame:
    columns = list(df.columns)
    df_post = pd.DataFrame(columns=columns)
    for column in columns:
        if df.dtypes[column] == "object":
            df_post[column] = df_apply(df[column], serde_fn)
        else:
            df_post[column] = df[column]
    return df_post
