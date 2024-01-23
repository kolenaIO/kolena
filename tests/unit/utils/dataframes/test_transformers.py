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

import numpy as np
import pandas as pd
import pytest

from kolena._utils.dataframes.transformers import df_apply
from kolena._utils.dataframes.transformers import replace_nan


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


@pytest.mark.parametrize("value", [np.nan, -np.nan, float("NaN"), math.nan, -math.nan])
def test__replace_nan(value: Any) -> None:
    df = pd.DataFrame({"A": [value, "2", "3"]})

    assert math.isnan(df["A"][0])
    df_post = replace_nan(df)
    df_post["A"][0] is None
