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
import uuid
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional

import pandas as pd

from tests.integration.conftest import TEST_PREFIX


def fake_locator(index: int, directory: str = "default") -> str:
    return f"https://fake-locator/{TEST_PREFIX}/{directory}/{index}.png"


def fake_random_locator(directory: str = "default") -> str:
    return f"https://fake-locator/{TEST_PREFIX}/{directory}/{uuid.uuid4()}.png"


def with_test_prefix(value: str) -> str:
    return f"{TEST_PREFIX} {value}"


def assert_sorted_list_equal(list_a: Iterable[Any], list_b: Iterable[Any]) -> None:
    assert sorted(list_a) == sorted(list_b)


def assert_frame_equal(df1: pd.DataFrame, df2: pd.DataFrame, columns: Optional[List[str]] = None) -> None:
    """wrapper of assert_frame_equal with selected columns options"""
    if columns is None:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=False)
    else:
        pd.testing.assert_frame_equal(df1[columns], df2[columns], check_dtype=False)
