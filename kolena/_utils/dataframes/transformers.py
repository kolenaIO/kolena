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
from collections.abc import Callable
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

import pandas as pd


def df_apply(df: pd.DataFrame, func: Callable, **kwargs: Any) -> pd.DataFrame:
    """
    pandas.DataFrame.apply wrapper
    """
    default_kwargs = dict(convert_dtype=False)
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


def json_normalize(data: List[Dict[str, Any]], max_level: int = 0) -> pd.DataFrame:
    """
    hand roll pandas.json_normalize implementation
    use None for missing value
    """
    normalized_data = _normalize_dicts(data, max_level=max_level)
    return pd.DataFrame.from_dict(normalized_data, dtype=object)
