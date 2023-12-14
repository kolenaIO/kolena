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
import pytest

from kolena._experimental.dataset.common import validate_id_fields


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
    df = pd.DataFrame(
        dict(
            locator=["http://1.jpg", "http://2.jpg", "http://3.jpg", "http://4.jpg"],
            locator2=["1", "1", "2", "2"],
            text=["a", "b", "c", "d"],
        ),
    )
    validate_id_fields(df, id_fields, existing_id_fields)


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
    df = pd.DataFrame(
        dict(
            locator=["http://1.jpg", "http://2.jpg", "http://3.jpg", "http://4.jpg"],
            locator2=["1", "1", "2", "2"],
            text=["a", "b", "c", "d"],
        ),
    )
    try:
        validate_id_fields(df, id_fields, existing_id_fields)
    except Exception as e:
        assert str(e) == expected_error
