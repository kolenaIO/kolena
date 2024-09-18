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
import re
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

    # Normalize all dictionaries to have all keys, filling missing ones with None
    for d in flattened_dicts:
        for key in all_keys:
            if key not in d:
                d[key] = None

    return flattened_dicts


def _try_parse(value: Any) -> Any:
    """
    Try parsing the input value into a more appropriate Python data type.
    This function handles the following conversions:

    - Empty strings are converted to None.
    - Strings that represent numbers are converted to numerical types (int or float).
    - JSON strings are converted to corresponding Python objects (lists or dictionaries).
    - Strings "true" and "false" are converted to their boolean equivalents.
    - NumPy arrays are converted to Python lists.
    - NaN values are converted to None.

    If none of the above conversions are applicable, the original value is returned.

    Parameters:
    value (Any): The input value to be parsed.

    Returns:
    Any: The parsed value if a conversion was successful, otherwise the original value.
    """

    # Handle NaN values
    if isinstance(value, float) and math.isnan(value):
        return None

    # Match behaviour: json.loads("NaN") == float('nan')
    if isinstance(value, str) and value == "NaN":
        return None

    # Convert empty strings to None
    if isinstance(value, str) and value == "":
        return None

    # Normalize NumPy arrays into Python lists to avoid downstream type errors
    if isinstance(value, np.ndarray):
        return value.tolist()

    # If the value is not a string, return it as is
    if not isinstance(value, str):
        return value

    # Convert strings "true" and "false" to boolean values
    if value.lower() in ["true", "false"]:
        return value.lower() == "true"

    # some string id could look like a scientific notation with e
    scientific_notation_pattern = re.compile(r"[+-]?\d+(\.\d+)?[eE][+-]?\d+")
    if scientific_notation_pattern.match(value):
        return value

    # Attempt to convert the string to a numeric type
    try:
        value_numeric = pd.to_numeric(value, errors="coerce")
        if not np.isnan(value_numeric):
            return value_numeric
    except Exception:
        pass

    # Only attempt to parse JSON arrays or objects
    if not (value.startswith("[") or value.startswith("{")):
        if value == "" or value == "null":
            return None
        elif value.startswith(("'", '"')) and value.endswith(("'", '"')):
            value = value[1:-1]
        return value

    # Attempt to parse the string as JSON
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
