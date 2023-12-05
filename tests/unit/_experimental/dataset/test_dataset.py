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

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from kolena._experimental.dataset._dataset import _infer_datatype
from kolena._experimental.dataset._dataset import _infer_datatype_value
from kolena._experimental.dataset._dataset import _to_deserialized_dataframe
from kolena._experimental.dataset._dataset import _to_serialized_dataframe
from kolena._experimental.dataset._dataset import add_datatype
from kolena._experimental.dataset._dataset import DatapointType
from kolena._experimental.dataset._evaluation import _align_datapoints_results
from kolena._experimental.dataset._evaluation import _validate_data
from kolena._experimental.dataset.common import COL_DATAPOINT
from kolena._experimental.dataset.common import COL_RESULT
from kolena.errors import IncorrectUsageError
from kolena.workflow._datatypes import DATA_TYPE_FIELD
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredClassificationLabel


@pytest.mark.parametrize(
    "uri,expected",
    [
        ("s3://public/png", DatapointType.TABULAR),
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
    add_datatype(df)
    assert df[DATA_TYPE_FIELD].equals(
        pd.Series([DatapointType.DOCUMENT, DatapointType.IMAGE, DatapointType.VIDEO, DatapointType.POINT_CLOUD]),
    )


def test__add_datatype__composite() -> None:
    def assert_datatype(dataset: pd.DataFrame, expected_datatype: DatapointType, prefix: str = "") -> None:
        for value in dataset[prefix + DATA_TYPE_FIELD]:
            assert value == expected_datatype

    composite_dataset = pd.DataFrame(
        {
            "a.text": [
                "A plane is taking off.",
                "A man is playing a large flute.",
                "A man is spreading shredded cheese on a pizza.",
            ],
        },
    )
    add_datatype(composite_dataset)
    assert_datatype(composite_dataset, DatapointType.COMPOSITE)
    assert_datatype(composite_dataset, DatapointType.TEXT, prefix="a.")

    composite_dataset["b.text"] = [
        "An air plane is taking off.",
        "A man is playing a flute.",
        "A man is spreading shredded cheese on an uncooked pizza.",
    ]
    composite_dataset["similarity"] = [5.0, 3.799999952316284, 3.799999952316284]
    add_datatype(composite_dataset)
    assert_datatype(composite_dataset, DatapointType.COMPOSITE)
    assert_datatype(composite_dataset, DatapointType.TEXT, prefix="a.")
    assert_datatype(composite_dataset, DatapointType.TEXT, prefix="b.")


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
                dict(text=dp["text"], value=dp["value"], category=dp["category"], data_type=DatapointType.TEXT)
                for dp in datapoints
            ],
        ),
    )
    df_serialized = _to_serialized_dataframe(df, column=COL_DATAPOINT)

    assert df_serialized[COL_DATAPOINT].apply(json.loads).equals(df_expected[COL_DATAPOINT])

    df_expected = pd.DataFrame(datapoints)
    df_deserialized = _to_deserialized_dataframe(df_serialized, column=COL_DATAPOINT)
    assert_frame_equal(df_deserialized, df_expected)


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

    df_expected = pd.DataFrame([["London"], ["Tokyo"], [np.nan]], columns=["city"])
    df_deserialized = _to_deserialized_dataframe(df_serialized, column=column_name)
    assert_frame_equal(df_deserialized, df_expected)


def test__dataframe__data_type_field_not_exist() -> None:
    column_name = COL_RESULT
    df_expected = pd.DataFrame([["a", "b", "c"], ["d", "e", "f"]])
    df_serialized = _to_serialized_dataframe(df_expected.copy(), column=column_name)
    assert column_name in df_serialized.columns
    for _, row in df_serialized.iterrows():
        assert DATA_TYPE_FIELD not in row[column_name]


def test__datapoints_results_alignment() -> None:
    df_datapoints = pd.DataFrame(dict(text=["a", "a", "b", "c"], question=["foo", "bar", "cat", "dog"]))
    df_results = pd.DataFrame(
        dict(text=["a", "a", "b"], question=["foo", "bar", "cat"], answer=[1, 2, 3]),
    )
    with pytest.raises(IncorrectUsageError):
        _align_datapoints_results(df_datapoints, df_results, on="text")

    df_merged = _align_datapoints_results(df_datapoints, df_results, on=["text", "question"])
    _validate_data(df_datapoints, df_merged)

    expected = pd.DataFrame(dict(answer=[1, 2, 3, np.nan]))
    assert df_merged.equals(expected)
