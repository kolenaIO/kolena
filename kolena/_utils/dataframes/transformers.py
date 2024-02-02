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

import numpy as np
import pandas as pd


def df_apply(df: pd.DataFrame, func: Callable, **kwargs: Any) -> pd.DataFrame:
    """
    pandas.DataFrame.apply wrapper
    """
    default_kwargs = dict(convert_dtype=False)
    return df.apply(func, **default_kwargs, **kwargs)


def replace_nan(df: pd.DataFrame) -> pd.DataFrame:
    df_post = df.astype("object").replace(np.nan, None)
    return df_post
