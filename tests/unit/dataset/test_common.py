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
from typing import List
from typing import Optional

import pandas as pd
import pytest

from kolena.dataset._common import validate_dataframe_columns
from kolena.dataset._common import validate_dataframe_ids
from kolena.dataset._common import validate_id_fields
from kolena.errors import DuplicateDatapointIdError
from kolena.errors import InputValidationError


@pytest.mark.parametrize(
    "id_fields, existing_id_fields",
    [
        (["locator"], ["locator"]),
        (["locator", "locator2"], ["locator", "locator2"]),
        (["locator"], []),
        (["locator", "locator2"], []),
    ],
)
def test__validate_id_fields__happy_path(id_fields: List[str], existing_id_fields: List[str]) -> None:
    validate_id_fields(id_fields, existing_id_fields)


@pytest.mark.parametrize(
    "id_fields, existing_id_fields, expected_error",
    [
        ([], ["locator1"], "invalid id_fields: expected at least one field"),
        (["locator", "locator"], [], "invalid id_fields: id fields should not contain duplicates"),
        (["locator3"], [], "invalid id_fields: field 'locator3' does not exist in dataframe"),
        (["locator2"], [], "invalid id_fields: input dataframe's id field values are not unique for ['locator2']"),
    ],
)
def test__validate_id_fields__validation_error(
    id_fields: List[str],
    existing_id_fields: List[str],
    expected_error: str,
) -> None:
    try:
        validate_id_fields(id_fields, existing_id_fields)
    except Exception as e:
        assert str(e) == expected_error


@pytest.mark.parametrize(
    "df, id_fields",
    [
        (pd.DataFrame(dict(a=[1, 2, 3], b=[1, 2, 1])), ["a", "b"]),
        (pd.DataFrame({"a.text": [1, 2, 3], "b.text": [1, 2, 1]}), ["a.text", "b.text"]),
        (pd.DataFrame(dict(a=[None, ""])), ["a"]),
        (pd.DataFrame(dict(a=[None, ""], b=[None, None])), ["a", "b"]),
    ],
)
def test__validate_dataframe_ids(df: pd.DataFrame, id_fields: List[str]) -> None:
    validate_dataframe_ids(df, id_fields)


@pytest.mark.parametrize(
    "df, id_fields",
    [
        # dataframe is missing one of id_fields
        (pd.DataFrame(dict(a=[1, 2, 3])), ["a", "b"]),
        # dataframe id fields are not hashable
        (pd.DataFrame(dict(a=[dict(c=i) for i in range(3)], b=[1, 2, 1])), ["a", "b"]),
        (pd.DataFrame(dict(a=[[1], [2], [3]], b=[1, 2, 1])), ["a", "b"]),
        # dataframe values in id_fields is not unique
        (pd.DataFrame(dict(a=[1, 2, 1], b=[1, 2, 1])), ["a", "b"]),
        # id columns not hashable
        (pd.DataFrame(dict(a=[{"a": 1, "b": 2}, {"a": 2, "b": 1}, {"b": 2, "a": 1}], b=[1, 2, 1])), ["a", "b"]),
    ],
)
def test__validate_dataframe_ids__error(df: pd.DataFrame, id_fields: List[str]) -> None:
    with pytest.raises(InputValidationError):
        validate_dataframe_ids(df, id_fields)


def test__validate_dataframe_ids__duplicate_id() -> None:
    with pytest.raises(DuplicateDatapointIdError) as e:
        data = list(range(15))
        a = data + data
        b = data + data[:12] + [1, 2, 3]
        # only report first 10
        expected = [dict(a=i, b=i) for i in range(10)]
        validate_dataframe_ids(pd.DataFrame(dict(a=a, b=b)), id_fields=["a", "b"])

    assert e.value.duplicate_ids == expected


@pytest.mark.parametrize(
    "df, id_fields, thresholded_fields",
    [
        (pd.DataFrame(dict(a=[1, 2, 3], b=[1, 2, 1])), ["a"], None),
        (pd.DataFrame({"a.text": [1, 2, 3], "b.text": [1, 2, 1]}), ["a.text"], None),
        (pd.DataFrame({"a.text": [1, 2, 3], "b.text": [1, 2, 1]}), ["a.text"], []),
        (
            pd.DataFrame({"a.text": [1, 2, 3], "b.text": [1, 2, 1], "threshold": [0.1, 0.2, 0.3]}),
            ["a.text"],
            ["threshold"],
        ),
    ],
)
def test__validate_dataframe_columns_besides_ids(
    df: pd.DataFrame,
    id_fields: List[str],
    thresholded_fields: Optional[List[str]],
) -> None:
    validate_dataframe_columns(df, id_fields, thresholded_fields)


@pytest.mark.parametrize(
    "df, id_fields, thresholded_fields",
    [
        (pd.DataFrame(dict(a=[1, 2, 3], b=[1, 2, 1])), ["a", "b"], None),
        (pd.DataFrame({"a.text": [1, 2, 3], "b.text": [1, 2, 1]}), ["a.text", "b.text"], None),
        (pd.DataFrame({"a.text": [1, 2, 3], "b.text": [1, 2, 1]}), ["a.text"], ["b.text"]),
    ],
)
def test__validate_dataframe_columns__error(
    df: pd.DataFrame,
    id_fields: List[str],
    thresholded_fields: Optional[List[str]],
) -> None:
    with pytest.raises(InputValidationError):
        validate_dataframe_columns(df, id_fields, thresholded_fields)
