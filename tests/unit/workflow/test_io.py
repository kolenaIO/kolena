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
from io import StringIO

import pandas as pd
from pandas._testing import assert_frame_equal

from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.io import dataframe_from_csv
from kolena.workflow.io import dataframe_from_json
from kolena.workflow.io import dataframe_to_csv

DF_TEST = pd.DataFrame.from_dict(
    {
        "id": list(range(10)),
        "z": [dict(value=i + 0.3) for i in range(10)],
        "partial": [None, ""] + ["fan"] * 8,
        "data": [
            LabeledBoundingBox(label=f"foo-{i}", top_left=[i, i], bottom_right=[i + 10, i + 10]) for i in range(10)
        ],
    },
)


def test__dataframe_json() -> None:
    json_str = DF_TEST.to_json()
    df_deserialized = dataframe_from_json(json_str)

    df_expected = pd.DataFrame.from_dict(
        {
            "id": list(range(10)),
            "z": [dict(value=i + 0.3) for i in range(10)],
            "partial": [None, ""] + ["fan"] * 8,
            "data": [BoundingBox(label=f"foo-{i}", top_left=[i, i], bottom_right=[i + 10, i + 10]) for i in range(10)],
        },
    )
    assert_frame_equal(df_deserialized, df_expected)
    assert df_deserialized.iloc[0]["data"].label == "foo-0"


def test__dataframe_csv() -> None:
    csv_str = dataframe_to_csv(DF_TEST, index=False)
    df_deserialized = dataframe_from_csv(StringIO(csv_str))

    df_expected = pd.DataFrame.from_dict(
        {
            "id": list(range(10)),
            "z": [dict(value=i + 0.3) for i in range(10)],
            "partial": [float("nan"), float("nan")] + ["fan"] * 8,
            "data": [BoundingBox(label=f"foo-{i}", top_left=[i, i], bottom_right=[i + 10, i + 10]) for i in range(10)],
        },
    )

    assert_frame_equal(df_deserialized, df_expected)
    assert df_deserialized.iloc[0]["id"] == 0
    assert df_deserialized.iloc[0]["data"].label == "foo-0"
