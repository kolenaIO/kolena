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
from typing import List

import pandas as pd

from kolena.errors import InputValidationError


COL_DATAPOINT = "datapoint"
COL_EVAL_CONFIG = "eval_config"
COL_RESULT = "result"


def validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise InputValidationError(f"invalid batch_size '{batch_size}': expected positive integer")


def validate_id_fields(df: pd.DataFrame, id_fields: List[str]) -> None:
    # TODO:
    # id fields is a set with a minimum length of 1
    # all id_fields exist in df
    # constrain the id fields type to be str or integer?
    ...
