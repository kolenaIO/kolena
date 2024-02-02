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
import math
from typing import Any

import pandas as pd

from kolena._utils.dataframes.transformers import df_apply
from kolena._utils.dataframes.transformers import json_normalize


def _try_convert_to_num(s: str) -> Any:
    if s == "":
        return None

    try:
        return pd.to_numeric(s)
    except Exception:
        return s


def test__df_apply() -> None:
    df = pd.DataFrame({"A": ["", "2", "3"]})

    assert df["A"][0] == ""
    df_post = df["A"].apply(_try_convert_to_num)
    assert math.isnan(df_post[0])
    # verify convert_dtype is False
    df_post = df_apply(df["A"], _try_convert_to_num)
    assert df_post[0] is None


def test__json_normalize():
    data = [
        {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
        {"name": {"given": "Mark", "family": "Regner"}},
        {"id": 3, "name": "Faye Raker"},
    ]
    normalized_data = [
        {"id": 1, "name": {"first": "Coleen", "last": "Volk"}},
        {"name": {"given": "Mark", "family": "Regner"}, "id": None},
        {"id": 3, "name": "Faye Raker"},
    ]
    df_expected = pd.DataFrame.from_dict(normalized_data, dtype=object)
    df_post = json_normalize(data)
    # assert the None is used for missing value
    assert df_post["id"][1] is None
    pd.testing.assert_frame_equal(df_post, df_expected)
