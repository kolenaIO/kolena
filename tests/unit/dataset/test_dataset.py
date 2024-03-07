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
import random

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kolena._api.v2.dataset import EntityData
from kolena._utils.datatypes import DATA_TYPE_FIELD
from kolena.dataset._common import COL_DATAPOINT
from kolena.dataset._common import COL_RESULT
from kolena.dataset.dataset import _add_datatype
from kolena.dataset.dataset import _infer_datatype
from kolena.dataset.dataset import _infer_datatype_value
from kolena.dataset.dataset import _infer_id_field
from kolena.dataset.dataset import _resolve_id_fields
from kolena.dataset.dataset import _to_deserialized_dataframe
from kolena.dataset.dataset import _to_serialized_dataframe
from kolena.dataset.dataset import DatapointType
from kolena.errors import InputValidationError
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredClassificationLabel


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("s3://public/png", DatapointType.TABULAR),
        ("s3://public/foo.png?versionId=42", DatapointType.IMAGE),
        ("s3://public/foo.pcd?bar=42#var", DatapointType.POINT_CLOUD),
        ("/opt/test.png", DatapointType.IMAGE),
        ("https://kolena.io/demo.mp4", DatapointType.VIDEO),
        ("file:///var/mime.csv", DatapointType.DOCUMENT),
        ("test.pcd", DatapointType.POINT_CLOUD),
        ("gcp://summary.pdf", DatapointType.DOCUMENT),
        ("//my.mp3", DatapointType.AUDIO),
    ],
)
def test__infer_datatype_value(uri: str, expected: str) -> None:
    assert _infer_datatype_value(uri) == expected


def test__add_datatype() -> None:
    df = pd.DataFrame(
        dict(
            locator=["s3://test.pdf", "https://test.png", "/home/test.mp4", "/tmp/test.pcd"],
        ),
    )
    _add_datatype(df)
    assert df[DATA_TYPE_FIELD].equals(
        pd.Series([DatapointType.DOCUMENT, DatapointType.IMAGE, DatapointType.VIDEO, DatapointType.POINT_CLOUD]),
    )


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
        == DatapointType.TABULAR
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
            label=ScoredClassificationLabel(label="dog", score=0.1 + i * 0.05),
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
                    label=dp["label"]._to_dict(),
                    data_type=DatapointType.IMAGE,
                )
                for dp in datapoints
            ],
        ),
    )
    df_serialized = _to_serialized_dataframe(df, column=COL_DATAPOINT)

    assert df_serialized[COL_DATAPOINT].apply(json.loads).equals(df_expected[COL_DATAPOINT])

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
                label=ClassificationLabel(label=dp["label"].label, score=dp["label"].score),
            )
            for dp in datapoints
        ],
    )
    df_deserialized = _to_deserialized_dataframe(df_serialized, column=COL_DATAPOINT)
    assert_frame_equal(df_deserialized, df_expected, check_dtype=False)


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
                dict(text=dp["text"], value=dp["value"], category=dp["category"], data_type=DatapointType.TEXT)
                for dp in datapoints
            ],
        ),
    )
    df_serialized = _to_serialized_dataframe(df, column=COL_DATAPOINT)

    assert df_serialized[COL_DATAPOINT].apply(json.loads).equals(df_expected[COL_DATAPOINT])

    df_expected = pd.DataFrame(datapoints)
    df_deserialized = _to_deserialized_dataframe(df_serialized, column=COL_DATAPOINT)
    assert_frame_equal(df_deserialized, df_expected, check_dtype=False)


def test__datapoint_dataframe__columns_unlabeled() -> None:
    df_expected = pd.DataFrame([["a", "b", "c"], ["d", "e", "f"]])
    df_serialized = _to_serialized_dataframe(df_expected.copy(), column=COL_DATAPOINT)
    df_deserialized = _to_deserialized_dataframe(df_serialized, column=COL_DATAPOINT)

    # Column class mismatch is expected due to json serialization
    df_expected.rename(mapper=str, axis="columns", inplace=True)
    assert_frame_equal(df_deserialized, df_expected)


def test__datapoint_dataframe__empty() -> None:
    df_serialized = _to_serialized_dataframe(pd.DataFrame(), column=COL_DATAPOINT)
    assert df_serialized.empty
    assert COL_DATAPOINT in df_serialized.columns


def test__datapoint_dataframe__data_type_field_exist() -> None:
    column_name = COL_DATAPOINT
    df_expected = pd.DataFrame([["a", "b", "c"], ["d", "e", "f"]])
    df_serialized = _to_serialized_dataframe(df_expected.copy(), column=column_name)
    assert column_name in df_serialized.columns
    for _, row in df_serialized.iterrows():
        assert DATA_TYPE_FIELD in row[column_name]


def test__dataframe__serde_none() -> None:
    column_name = COL_RESULT
    data = [
        ['{"city": "London"}'],
        ['{"city": "Tokyo"}'],
        [None],
    ]
    df_serialized = pd.DataFrame(data, columns=[column_name])

    df_expected = pd.DataFrame([["London"], ["Tokyo"], [None]], columns=["city"])
    df_deserialized = _to_deserialized_dataframe(df_serialized, column=column_name)
    assert_frame_equal(df_deserialized, df_expected)


def test__dataframe__data_type_field_not_exist() -> None:
    column_name = COL_RESULT
    df_expected = pd.DataFrame([["a", "b", "c"], ["d", "e", "f"]])
    df_serialized = _to_serialized_dataframe(df_expected.copy(), column=column_name)
    assert column_name in df_serialized.columns
    for _, row in df_serialized.iterrows():
        assert DATA_TYPE_FIELD not in row[column_name]


@pytest.mark.parametrize(
    "input_df, expected",
    [
        (
            pd.DataFrame(
                dict(
                    id=[1, 2, 3, 4],
                    locator=["s3://test.pdf", "https://test.png", "/home/test.mp4", "/tmp/test.pcd"],
                ),
            ),
            "id",
        ),
        (
            pd.DataFrame(
                dict(
                    locator=["s3://test.pdf", "https://test.png", "/home/test.mp4", "/tmp/test.pcd"],
                    text=["a", "b", "c", "d"],
                ),
            ),
            "locator",
        ),
        (
            pd.DataFrame(
                dict(
                    text=["a", "b", "c", "d"],
                ),
            ),
            "text",
        ),
    ],
)
def test__infer_id_field(input_df: pd.DataFrame, expected: str) -> None:
    assert _infer_id_field(input_df) == expected


@pytest.mark.parametrize(
    "input_df",
    [
        pd.DataFrame(dict(text1=["a", "b", "c", "d"])),
    ],
)
def test__infer_id_field__error(input_df: pd.DataFrame) -> None:
    with pytest.raises(Exception) as e:
        _infer_id_field(input_df)

    error_msg = "Failed to infer the id_fields, please provide id_fields explicitly"
    assert str(e.value) == error_msg


def test__resolve_id_fields() -> None:
    df = pd.DataFrame(dict(user_dp_id=["a", "b", "c"], new_user_dp_id=["d", "e", "f"]))
    dataset = EntityData(id=1, name="foo", description="", id_fields=["user_dp_id"])
    inferrable_df = pd.DataFrame(dict(locator=["x", "y", "z"]))

    # new dataset without id_fields
    with pytest.raises(InputValidationError):
        _resolve_id_fields(df, None, None)

    # existing dataset without id_fields, different inferred id_fields, should use existing id_fields
    assert _resolve_id_fields(inferrable_df, None, dataset) == ["user_dp_id"]

    # existing dataset without id_fields, same inferred id_fields
    assert _resolve_id_fields(
        inferrable_df,
        None,
        EntityData(id=1, name="foo", description="", id_fields=["locator"]),
    ) == ["locator"]

    # new dataset with explicit id_fields should resolve to explicit id_fields
    assert _resolve_id_fields(df, ["user_dp_id"], None) == ["user_dp_id"]

    # existing dataset id_fields are the same as explicit id_fields
    assert _resolve_id_fields(df, ["user_dp_id"], dataset) == ["user_dp_id"]

    # explicit id_fields override existing dataset id_fields
    assert _resolve_id_fields(df, ["new_user_dp_id"], dataset) == ["new_user_dp_id"]

    # new dataset with implicit datatype support, e.g. locator, without id_fields
    assert _resolve_id_fields(inferrable_df, None, None) == ["locator"]
