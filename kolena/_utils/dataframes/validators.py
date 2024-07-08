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
import re
from typing import Optional
from typing import Type
from typing import TypeVar

import pandas as pd
import pandera as pa
from pandera.errors import SchemaError
from pandera.extensions import register_check_method
from pandera.typing import Series

from kolena.errors import InputValidationError


T = TypeVar("T", bound=pa.DataFrameModel)


def validate_df_schema(df: pd.DataFrame, schema: Type[T], trusted: bool = False) -> pd.DataFrame:
    """
    Validate the provided DataFrame against the schema, applying type coercions to all cells in-place and explicitly
    validating up to 1,000 rows for "trusted" frames (i.e. assembled by us) and all rows for "untrusted" frames
    (i.e. provided by the user).
    """
    sample_size = 1000
    sample_kwargs = dict(head=10, sample=sample_size, tail=10) if trusted and len(df) > sample_size else {}
    kwargs = dict(inplace=True, **sample_kwargs)
    try:
        return schema.validate(df, **kwargs)
    except SchemaError as e:
        raise InputValidationError(e)


def validate_df_record_count(df: pd.DataFrame, max_records_allowed: Optional[int] = None) -> None:
    if len(df) == 0:
        raise InputValidationError("zero records provided")
    if max_records_allowed is not None and len(df) > max_records_allowed:
        raise InputValidationError(
            f"Too many records provided to upload at once: got {len(df)}, max {max_records_allowed} allowed per call. "
            "Consider splitting the input via e.g. `np.array_split(...)`.",
        )


# Share Registered check methods for validation
# https://pandera.readthedocs.io/en/stable/reference/generated/pandera.extensions.html?highlight=register_check_method#pandera.extensions.register_check_method


def _is_locator_cell_valid(cell: pa.typing.String) -> bool:
    matcher = re.compile(r"^(s3|gs|https)://(.+/)+.+$")
    return isinstance(cell, str) and bool(matcher.match(cell.lower()))


def _is_image_locator_cell_valid(cell: pa.typing.String) -> bool:
    matcher = re.compile(r"^(s3|gs|https)://(.+/)+.+\.(png|jpe?g|gif|tiff?)$")
    return isinstance(cell, str) and bool(matcher.match(cell.lower()))


@register_check_method(check_type="element_wise")
def _element_wise_validate_locator(cell: pa.typing.String) -> bool:
    return _is_locator_cell_valid(cell)


@register_check_method(check_type="element_wise")
def _element_wise_validate_image_locator(cell: pa.typing.String) -> bool:
    return _is_image_locator_cell_valid(cell)


@register_check_method()
def _validate_locator(series: Series) -> bool:
    return series.dropna().apply(_is_locator_cell_valid).all()


@register_check_method()
def _validate_image_locator(series: Series) -> bool:
    return series.dropna().apply(_is_image_locator_cell_valid).all()
