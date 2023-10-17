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
import math
import random

import pandas as pd
import pytest
from pandas._testing import assert_frame_equal
from pandas._testing import assert_series_equal

from kolena._experimental.dataset._dataset import _consolidate_field_order
from kolena._experimental.dataset._dataset import _infer_datatype
from kolena._experimental.dataset._dataset import _infer_datatype_value
from kolena._experimental.dataset._dataset import _to_deserialized_dataframe
from kolena._experimental.dataset._dataset import _to_serialized_dataframe
from kolena._experimental.dataset._dataset import COL_DATAPOINT
from kolena._experimental.dataset._dataset import DatapointType
from kolena.workflow._datatypes import FIELD_ORDER_FIELD
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredClassificationLabel


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("s3://public/png", DatapointType.CUSTOM),
        ("/opt/test.png", DatapointType.IMAGE),
        ("https://kolena.io/demo.mp4", DatapointType.VIDEO),
        ("file:///var/mime.csv", DatapointType.DOCUMENT),
        ("test.pcd", DatapointType.POINT_CLOUD),
        ("gcp://summary.pdf", DatapointType.DOCUMENT),
        ("//my.mp3", DatapointType.CUSTOM),
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
    ).equals(
        pd.Series([DatapointType.DOCUMENT, DatapointType.IMAGE, DatapointType.VIDEO, DatapointType.POINT_CLOUD]),
    )
    assert _infer_datatype(
        pd.DataFrame(
            dict(
                locator=["s3://test.pdf", "https://test.png", "/home/test.mp4", "/tmp/test.pcd"],
                text=["a", "b", "c", "d"],
            ),
        ),
    ).equals(
        pd.Series([DatapointType.DOCUMENT, DatapointType.IMAGE, DatapointType.VIDEO, DatapointType.POINT_CLOUD]),
    )
    assert (
        _infer_datatype(
            pd.DataFrame(
                dict(
                    text=["a", "b", "c", "d"],
                ),
            ),
        )
        == DatapointType.TEXT
    )
    assert (
        _infer_datatype(
            pd.DataFrame(
                dict(
                    category=["a", "b", "c", "d"],
                ),
            ),
        )
        == DatapointType.CUSTOM
    )


def test__consolidate_field_order() -> None:
    datapoints = [
        {FIELD_ORDER_FIELD: ["b", "c", "a"]},
        {FIELD_ORDER_FIELD: ["a", "b", "d"]},
        {FIELD_ORDER_FIELD: ["y", "d", "c", "x"]},
    ]
    assert _consolidate_field_order(datapoints) == ["b", "c", "a", "d", "y", "x"]


def test__datapoint_dataframe__serde_locator() -> None:
    # emulate mixing datapoints of different structure, as is possible from dataset updates/curations
    datapoints_v1 = [
        dict(
            locator=f"https://test-iamge-{i}.png",
            width=500 + i,
            height=400 + i,
            category="outdoor" if i < 5 else "indoor",
            bboxes=[
                LabeledBoundingBox(label="car", top_left=[i, i], bottom_right=[i + 50, i + 50])
                for i in range(random.randint(2, 6))
            ],
            label=ScoredClassificationLabel(label="dog", score=0.1 + i * 0.05),
        )
        for i in range(8)
    ]
    datapoints_v2 = [
        dict(
            locator=f"https://test-iamge-{i}.png",
            bboxes=[
                LabeledBoundingBox(label="car", top_left=[i, i], bottom_right=[i + 50, i + 50])
                for i in range(random.randint(2, 6))
            ],
            x=f"v{i}",
            width=500 + i,
        )
        for i in range(8, 10)
    ]
    df_v1 = pd.DataFrame(datapoints_v1)
    df_v2 = pd.DataFrame(datapoints_v2)
    df_expected = pd.DataFrame(
        dict(
            datapoint=[
                dict(
                    locator=dp["locator"],
                    width=dp["width"],
                    height=dp["height"],
                    category=dp["category"],
                    bboxes=[bbox._to_dict() for bbox in dp["bboxes"]],
                    label=dp["label"]._to_dict(),
                    data_type=DatapointType.IMAGE,
                    _field_order=["locator", "width", "height", "category", "bboxes", "label"],
                )
                for dp in datapoints_v1
            ]
            + [
                dict(
                    locator=dp["locator"],
                    bboxes=[bbox._to_dict() for bbox in dp["bboxes"]],
                    x=dp["x"],
                    width=dp["width"],
                    data_type=TestSampleType.IMAGE,
                    _field_order=["locator", "bboxes", "x", "width"],
                )
                for dp in datapoints_v2
            ],
        ),
    )
    df_serialized = pd.concat([_to_serialized_dataframe(df_v1), _to_serialized_dataframe(df_v2)], ignore_index=True)

    assert_series_equal(df_serialized[COL_DATAPOINT].apply(json.loads), df_expected[COL_DATAPOINT])

    df_expected = pd.DataFrame(
        [
            dict(
                locator=dp["locator"],
                width=dp["width"],
                height=dp.get("height", math.nan),
                category=dp.get("category", math.nan),
                bboxes=[
                    BoundingBox(label=bbox.label, top_left=bbox.top_left, bottom_right=bbox.bottom_right)
                    for bbox in dp["bboxes"]
                ],
                label=ClassificationLabel(label=dp["label"].label, score=dp["label"].score)
                if "label" in dp
                else math.nan,
                x=dp.get("x", math.nan),
            )
            for dp in datapoints_v1 + datapoints_v2
        ],
    )
    df_deserialized = _to_deserialized_dataframe(df_serialized)
    assert_frame_equal(df_deserialized, df_expected)


def test__datapoint_dataframe__serde_text() -> None:
    datapoints = [
        dict(
            text=f"foo-{i}",
            value=i,
            category="A" if i < 5 else "B",
        )
        for i in range(10)
    ]
    df = pd.DataFrame(datapoints)
    df_expected = pd.DataFrame(
        dict(
            datapoint=[
                dict(
                    text=dp["text"],
                    value=dp["value"],
                    category=dp["category"],
                    data_type=DatapointType.TEXT,
                    _field_order=["text", "value", "category"],
                )
                for dp in datapoints
            ],
        ),
    )
    df_serialized = _to_serialized_dataframe(df)

    assert df_serialized[COL_DATAPOINT].apply(json.loads).equals(df_expected[COL_DATAPOINT])

    df_expected = pd.DataFrame(datapoints)
    df_deserialized = _to_deserialized_dataframe(df_serialized)
    assert_frame_equal(df_deserialized, df_expected)


def test__datapoint_dataframe__columns_unlabeled() -> None:
    df_expected = pd.DataFrame([["a", "b", "c"], ["d", "e", "f"]])
    df_serialized = _to_serialized_dataframe(df_expected.copy())
    df_deserialized = _to_deserialized_dataframe(df_serialized)

    # Column class mismatch is expected due to json serialization
    df_expected.rename(mapper=str, axis="columns", inplace=True)
    assert_frame_equal(df_deserialized, df_expected)


def test__datapoint_dataframe__empty() -> None:
    df_serialized = _to_serialized_dataframe(pd.DataFrame())
    assert df_serialized.empty
    assert COL_DATAPOINT in df_serialized.columns
