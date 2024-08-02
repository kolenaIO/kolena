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
import json
import uuid
from typing import Any
from typing import Iterable
from typing import List
from typing import Optional

import pandas as pd

from kolena._utils import krequests
from kolena._utils.batched_load import init_upload
from kolena._utils.batched_load import upload_data_frame
from kolena.dataset._common import COL_DATAPOINT_ID_OBJECT
from kolena.dataset.dataset import _to_serialized_dataframe
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


def upload_extracted_properties(
    dataset_id: int,
    df: pd.DataFrame,
    id_fields: list[str],
    property_type: str = "llm",
    property_source: str = "datapoint",
) -> pd.DataFrame:
    df_serialized_id_object = _to_serialized_dataframe(
        df[sorted(id_fields)],
        column=COL_DATAPOINT_ID_OBJECT,
    )
    df = pd.concat([df, df_serialized_id_object], axis=1)
    init_response = init_upload()
    upload_data_frame(df=df, load_uuid=init_response.uuid)
    response = krequests.post(
        "/search/extracted-properties",
        json.dumps(
            dict(
                uuid=init_response.uuid,
                id=dataset_id,
                property_type=property_type,
                property_source=property_source,
            ),
        ),
        api_version="v2",
    )
    return response
