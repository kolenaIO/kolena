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
import math
import warnings
from collections.abc import Callable
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import numpy as np
import pandas as pd


def df_apply(df: pd.DataFrame, func: Callable, **kwargs: Any) -> pd.DataFrame:
    """
    pandas.DataFrame.apply wrapper
    """
    default_kwargs = dict(convert_dtype=False)
    with warnings.catch_warnings():
        # Silence the warning for now -- pandas devs have deprecated this without providing a good alternative:
        # https://github.com/pandas-dev/pandas/pull/52257#issuecomment-1684888371
        warnings.simplefilter(action="ignore", category=FutureWarning)
        return df.apply(func, **default_kwargs, **kwargs)


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".", max_level: int = 0) -> Dict[str, Any]:
    items: List[Tuple] = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict) and max_level > 0:
            items.extend(_flatten_dict(v, new_key, sep, max_level - 1).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _normalize_dicts(dict_list: List[Dict[str, Any]], sep: str = ".", max_level: int = 0) -> List[Dict[str, Any]]:
    # Flatten all dictionaries in the list
    flattened_dicts = [_flatten_dict(d, sep=sep, max_level=max_level) for d in dict_list]

    # Get all unique keys across all dictionaries
    all_keys = set().union(*flattened_dicts)

    # Normalize all dictionaries to have all keys,
    # filling missing ones with None and parsing existing ones
    for d in flattened_dicts:
        for key in all_keys:
            d[key] = _try_parse(d.get(key, None))

    return flattened_dicts


def _try_parse(value: Any) -> Any:
    """
    Try parsing:
        - empty string as None
        - numbers
        - a JSON string into a Python object.
    Return parsed value if successful, otherwise the original value.
    """
    # handle na
    if isinstance(value, float) and math.isnan(value):
        return None

    # normalize np.ndarray into python list to avoid downstream type error:
    # Object of type ndarray is not JSON serializable
    if isinstance(value, np.ndarray):
        return value.tolist()

    if not isinstance(value, str):
        return value

    if value == "":
        return None

    try:
        return pd.to_numeric(value)
    except Exception:
        ...

    # only parse json array or object
    if not (value.startswith("[") or value.startswith("{")):
        return value
    try:
        return json.loads(value)
    except Exception:
        return value


def json_normalize(data: List[Dict[str, Any]], max_level: int = 0) -> pd.DataFrame:
    """
    hand roll pandas.json_normalize implementation
    use None for missing value
    """
    normalized_data = _normalize_dicts(data, max_level=max_level)
    return pd.DataFrame.from_dict(normalized_data, dtype=object)


def parse_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalize strings in DataFrame columns into respective Python objects.
    """
    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].apply(lambda x: _try_parse(x), convert_dtype=False)
    return df


def drop_unnamed(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops any columns in the DataFrame that have an 'Unnamed' prefix in their column name.
    """
    cols_to_drop = [col for col in df.columns if col.startswith("Unnamed")]
    df.drop(columns=cols_to_drop, inplace=True)
    return df


def drop_invalid_ground_truth(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the invalid 'ground_truth' column from the DataFrame.
    """
    GROUND_TRUTH_FIELD = "ground_truth"
    if GROUND_TRUTH_FIELD in df.columns:
        if not isinstance(df[GROUND_TRUTH_FIELD].iloc[0], dict):
            df.drop(columns=[GROUND_TRUTH_FIELD], inplace=True)
    return df


def drop_invalid_metadata(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the invalid 'metadata' column from the DataFrame.
    """
    METADATA_FIELD = "metadata"
    if METADATA_FIELD in df.columns:
        if not isinstance(df[METADATA_FIELD].iloc[0], dict):
            df.drop(columns=[METADATA_FIELD], inplace=True)
    return df
