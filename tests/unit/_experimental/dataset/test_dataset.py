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
import json
import random

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal

from kolena._experimental.dataset._dataset import _infer_datatype
from kolena._experimental.dataset._dataset import _infer_datatype_value
from kolena._experimental.dataset._dataset import _to_deserialized_dataframe
from kolena._experimental.dataset._dataset import _to_serialized_dataframe
from kolena._experimental.dataset._dataset import TEST_SAMPLE_TYPE
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import LabeledBoundingBox

CUSTOM = f"{TEST_SAMPLE_TYPE}/CUSTOM"
IMAGE = f"{TEST_SAMPLE_TYPE}/IMAGE"
VIDEO = f"{TEST_SAMPLE_TYPE}/VIDEO"
DOCUMENT = f"{TEST_SAMPLE_TYPE}/DOCUMENT"
TEXT = f"{TEST_SAMPLE_TYPE}/TEXT"
POINT_CLOUD = f"{TEST_SAMPLE_TYPE}/POINT_CLOUD"


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("s3://public/png", CUSTOM),
        ("/opt/test.png", IMAGE),
        ("https://kolena.io/demo.mp4", VIDEO),
        ("file:///var/mime.csv", DOCUMENT),
        ("test.pcd", POINT_CLOUD),
        ("gcp://summary.pdf", DOCUMENT),
        ("//my.mp3", CUSTOM),
    ],
)
def test__infer_datatype_value(uri: str, expected: str) -> None:
    assert _infer_datatype_value(uri) == expected


def test__infer_datatype() -> None:
    assert _infer_datatype(
        pd.DataFrame(
            dict(
                locator=["s3://test.pdf", "https://test.png", "/home/test.mp4", "/tmp/test.pcd"],
            ),
        ),
    ).equals(pd.Series([DOCUMENT, IMAGE, VIDEO, POINT_CLOUD]))
    assert _infer_datatype(
        pd.DataFrame(
            dict(
                locator=["s3://test.pdf", "https://test.png", "/home/test.mp4", "/tmp/test.pcd"],
                text=["a", "b", "c", "d"],
            ),
        ),
    ).equals(pd.Series([DOCUMENT, IMAGE, VIDEO, POINT_CLOUD]))
    assert (
        _infer_datatype(
            pd.DataFrame(
                dict(
                    text=["a", "b", "c", "d"],
                ),
            ),
        )
        == TEXT
    )
    assert (
        _infer_datatype(
            pd.DataFrame(
                dict(
                    category=["a", "b", "c", "d"],
                ),
            ),
        )
        == CUSTOM
    )


def test__datapoint_dataframe__serde_locator() -> None:
    datapoints = [
        dict(
            locator=f"https://test-iamge-{i}.png",
            width=500 + i,
            height=400 + i,
            category="outdoor" if i < 5 else "indoor",
            bboxes=[
                LabeledBoundingBox(label="car", top_left=[i, i], bottom_right=[i + 50, i + 50])
                for i in range(random.randint(2, 6))
            ],
        )
        for i in range(10)
    ]
    df = pd.DataFrame(datapoints)
    df_expected = pd.DataFrame(
        dict(
            datapoint=[
                dict(
                    locator=dp["locator"],
                    width=dp["width"],
                    height=dp["height"],
                    category=dp["category"],
                    bboxes=[bbox._to_dict() for bbox in dp["bboxes"]],
                    data_type=IMAGE,
                )
                for dp in datapoints
            ],
        ),
    )
    df_serialized = _to_serialized_dataframe(df)

    assert df_serialized["datapoint"].apply(json.loads).equals(df_expected["datapoint"])

    df_expected = pd.DataFrame(
        [
            dict(
                locator=dp["locator"],
                width=dp["width"],
                height=dp["height"],
                category=dp["category"],
                bboxes=[
                    BoundingBox(label=bbox.label, top_left=bbox.top_left, bottom_right=bbox.bottom_right)
                    for bbox in dp["bboxes"]
                ],
            )
            for dp in datapoints
        ],
    )
    df_deserialized = _to_deserialized_dataframe(df_serialized)
    assert sorted(df_deserialized.columns) == sorted(df_expected.columns)

    assert_frame_equal(df_deserialized[df_expected.columns], df_expected)


def test__datapoint_dataframe__serde_text() -> None:
    datapoints = [
        dict(
            text=f"foo-{i}",
            category="A" if i < 5 else "B",
        )
        for i in range(10)
    ]
    df = pd.DataFrame(datapoints)
    df_expected = pd.DataFrame(
        dict(
            datapoint=[dict(text=dp["text"], category=dp["category"], data_type=TEXT) for dp in datapoints],
        ),
    )
    df_serialized = _to_serialized_dataframe(df)

    assert df_serialized["datapoint"].apply(json.loads).equals(df_expected["datapoint"])

    df_expected = pd.DataFrame([dict(text=dp["text"], category=dp["category"]) for dp in datapoints])
    df_deserialized = _to_deserialized_dataframe(df_serialized)
    assert sorted(df_deserialized.columns) == sorted(df_expected.columns)

    assert_frame_equal(df_deserialized[df_expected.columns], df_expected)
