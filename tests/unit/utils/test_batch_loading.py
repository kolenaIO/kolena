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
from unittest.mock import patch

import pandas as pd

from kolena._utils.batched_load import upload_data_frame

TEST_UUID = "test uuid"
MOCK_MEMORY_SIZE = 3 * pow(1024, 3)
MOCK_FILE_SIZE = 100 * pow(1024, 2)


def test__file_short_circut() -> None:
    height = 10
    width = 20
    df = pd.DataFrame(0, index=range(height), columns=range(width))

    with patch("kolena._utils.batched_load._upload_data_frame") as upload_helper:
        upload_data_frame(df, TEST_UUID)

    upload_helper.assert_called_once_with(df, len(df), TEST_UUID)


def test__file_over_short_circut() -> None:
    height = 5000
    width = 20
    df = pd.DataFrame(0, index=range(height), columns=range(width))

    with patch("kolena._utils.batched_load._upload_data_frame") as upload_helper:
        with patch("kolena._utils.batched_load._calculate_memory_size") as memory_size:
            with patch("kolena._utils.batched_load._get_preflight_export_size") as preflight_size:
                memory_size.return_value = MOCK_MEMORY_SIZE
                preflight_size.return_value = MOCK_FILE_SIZE
                upload_data_frame(df, TEST_UUID)

    # batch size is 2000 rows since we are using // in the calculation
    upload_helper.assert_called_once_with(df, 2000, TEST_UUID)


def test__file_over_short_circut_verify_chunks() -> None:
    height = 5000
    width = 20
    df = pd.DataFrame(0, index=range(height), columns=range(width))

    with patch("kolena._utils.batched_load.upload_data_frame_chunk") as upload_chunk:
        with patch("kolena._utils.batched_load._calculate_memory_size") as memory_size:
            with patch("kolena._utils.batched_load._get_preflight_export_size") as preflight_size:
                memory_size.return_value = MOCK_MEMORY_SIZE
                preflight_size.return_value = MOCK_FILE_SIZE
                upload_data_frame(df, TEST_UUID)

    assert upload_chunk.call_count == 3
