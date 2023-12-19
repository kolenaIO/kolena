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
from collections import Counter
from typing import List

import pandas as pd

from kolena._utils import log
from kolena.errors import InputValidationError

COL_DATAPOINT = "datapoint"
COL_DATAPOINT_ID_OBJECT = "datapoint_id_object"
COL_EVAL_CONFIG = "eval_config"
COL_RESULT = "result"


def validate_batch_size(batch_size: int) -> None:
    if batch_size <= 0:
        raise InputValidationError(f"invalid batch_size '{batch_size}': expected positive integer")


def validate_id_fields(id_fields: List[str], existing_id_fields: List[str] = None) -> None:
    if len(id_fields) == 0:
        raise InputValidationError("invalid id_fields: expected at least one field")
    if len(Counter(id_fields)) != len(id_fields):
        raise InputValidationError(
            "invalid id_fields: id fields should not contain duplicates",
        )
    if existing_id_fields:
        if set(id_fields) != set(existing_id_fields):
            log.warn(
                f"ID field for the existing dataset has been changed from {existing_id_fields} to {id_fields},"
                f" this will disassociate the existing result from the new datapoints",
            )


def _validate_dataframe_ids_uniqueness(df: pd.DataFrame, id_fields: List[str]) -> None:
    if df[id_fields].apply(lambda x: x.to_dict(), axis=1).duplicated().any():
        raise InputValidationError(
            f"invalid id_fields: " f"input dataframe's id field values are not unique for {id_fields}",
        )


def validate_dataframe_ids(df: pd.DataFrame, id_fields: List[str]) -> None:
    for id_field in id_fields:
        if id_field not in df.columns:
            raise InputValidationError(
                f"invalid id_fields: field '{id_field}' does not exist in dataframe",
            )
    _validate_dataframe_ids_uniqueness(df, id_fields)
