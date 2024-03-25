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
from io import StringIO
from math import isnan

import pandas as pd
from pandas.testing import assert_frame_equal

from kolena.annotation import BoundingBox
from kolena.annotation import LabeledBoundingBox
from kolena.io import _serialize_dataobject_str
from kolena.io import dataframe_from_csv
from kolena.io import dataframe_from_json
from kolena.io import dataframe_to_csv

NAN = float("nan")
DF_TEST = pd.DataFrame.from_dict(
    {
        "z": [dict(value=i + 0.3) for i in range(10)],
        "partial": [None, ""] + ["fan"] * 8,
        "data": [
            LabeledBoundingBox(label=f"foo-{i}", top_left=[i, i], bottom_right=[i + 10, i + 10]) for i in range(10)
        ],
        "deep_data": [
            dict(asset=[LabeledBoundingBox(label=f"bar-{i}", top_left=[i, i], bottom_right=[i + 10, i + 10])])
            for i in range(10)
        ],
        "id": list(range(10)),
        "bad actor": [
            "{",
            dict(value="box"),
            15,
            None,
            "foo",
            [1, "3", "5"],
            LabeledBoundingBox(label="cat", top_left=[3, 5], bottom_right=[10, 15]),
            "",
        ]
        + ["car"] * 2,
    },
)


def test__dataframe_json() -> None:
    json_str = DF_TEST.to_json()
    df_deserialized = dataframe_from_json(json_str)

    json_df_expected = pd.DataFrame.from_dict(
        {
            "z": [dict(value=i + 0.3) for i in range(10)],
            "partial": [None, ""] + ["fan"] * 8,
            "data": [BoundingBox(label=f"foo-{i}", top_left=[i, i], bottom_right=[i + 10, i + 10]) for i in range(10)],
            "deep_data": [
                dict(asset=[BoundingBox(label=f"bar-{i}", top_left=[i, i], bottom_right=[i + 10, i + 10])])
                for i in range(10)
            ],
            "id": list(range(10)),
            "bad actor": [
                "{",
                dict(value="box"),
                15,
                None,
                "foo",
                [1, "3", "5"],
                BoundingBox(label="cat", top_left=[3, 5], bottom_right=[10, 15]),
                "",
            ]
            + ["car"] * 2,
        },
    )

    assert_frame_equal(df_deserialized, json_df_expected)
    assert df_deserialized.iloc[0]["id"] == 0
    assert df_deserialized.iloc[0]["data"].label == "foo-0"


def test__dataframe_csv() -> None:
    csv_str = dataframe_to_csv(DF_TEST, index=False)
    df_deserialized = dataframe_from_csv(StringIO(csv_str))

    csv_df_expected = pd.DataFrame.from_dict(
        {
            "z": [dict(value=i + 0.3) for i in range(10)],
            "partial": [NAN, NAN] + ["fan"] * 8,
            "data": [BoundingBox(label=f"foo-{i}", top_left=[i, i], bottom_right=[i + 10, i + 10]) for i in range(10)],
            "deep_data": [
                dict(asset=[BoundingBox(label=f"bar-{i}", top_left=[i, i], bottom_right=[i + 10, i + 10])])
                for i in range(10)
            ],
            "id": list(range(10)),
            "bad actor": [
                "{",
                dict(value="box"),
                15,
                NAN,
                "foo",
                [1, "3", "5"],
                BoundingBox(label="cat", top_left=[3, 5], bottom_right=[10, 15]),
                NAN,
            ]
            + ["car"] * 2,
        },
    )

    assert_frame_equal(df_deserialized, csv_df_expected)
    assert df_deserialized.iloc[0]["id"] == 0
    assert df_deserialized.iloc[0]["data"].label == "foo-0"


def test___serialize_dataobject_str() -> None:
    # does not serialize
    assert 1 == _serialize_dataobject_str(1)
    assert "locator" == _serialize_dataobject_str("locator")
    assert isnan(_serialize_dataobject_str(NAN))

    labeled_bbox = LabeledBoundingBox(label="foo", top_left=[0, 0], bottom_right=[10, 10])
    bbox = BoundingBox(label="foo", top_left=[0, 0], bottom_right=[10, 10])

    # serializes
    assert json.dumps(bbox._to_dict()) == _serialize_dataobject_str(labeled_bbox)
    assert json.dumps(["locator"]) == _serialize_dataobject_str(["locator"])
    assert json.dumps(dict(field1=True, field2="locator", field3=bbox._to_dict())) == _serialize_dataobject_str(
        dict(field1=True, field2="locator", field3=labeled_bbox),
    )


def test__dataframe_csv__malformed_input() -> None:
    input_str = "A,B,C,D\n" """1,"foo","[""a"",""b"",""c""]",false\n""" """2,"{bar}","[1,2,3",tru\n""" """,,,\n"""
    df = dataframe_from_csv(StringIO(input_str))
    df_expected = pd.DataFrame(
        dict(
            A=[1, 2, NAN],
            B=["foo", "{bar}", NAN],
            C=[["a", "b", "c"], "[1,2,3", NAN],
            D=[False, "tru", NAN],
        ),
    )
    assert_frame_equal(df, df_expected)
