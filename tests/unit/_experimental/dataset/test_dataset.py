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

from .data import a_text
from .data import b_text
from kolena._experimental.dataset._dataset import _add_datatype
from kolena._experimental.dataset._dataset import _flatten_composite
from kolena._experimental.dataset._dataset import _infer_datatype
from kolena._experimental.dataset._dataset import _infer_datatype_value
from kolena._experimental.dataset._dataset import _infer_id_fields
from kolena._experimental.dataset._dataset import _to_deserialized_dataframe
from kolena._experimental.dataset._dataset import _to_serialized_dataframe
from kolena._experimental.dataset._dataset import DatapointType
from kolena._experimental.dataset._evaluation import _align_datapoints_results
from kolena._experimental.dataset._evaluation import _validate_data
from kolena._experimental.dataset.common import COL_DATAPOINT
from kolena._experimental.dataset.common import COL_RESULT
from kolena.errors import IncorrectUsageError
from kolena.errors import InputValidationError
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
    _add_datatype(df)
    assert df[DATA_TYPE_FIELD].equals(
        pd.Series([DatapointType.DOCUMENT, DatapointType.IMAGE, DatapointType.VIDEO, DatapointType.POINT_CLOUD]),
    )


def test__add_datatype__composite() -> None:
    composite_dataset = pd.DataFrame(
        {
            "a.text": a_text,
        },
    )
    _add_datatype(composite_dataset)

    assert "a.text" not in composite_dataset
    assert (composite_dataset[DATA_TYPE_FIELD] == DatapointType.COMPOSITE).all()
    assert (composite_dataset["a"] == [{"text": text, DATA_TYPE_FIELD: DatapointType.TEXT} for text in a_text]).all()

    similarity = [5.0, 3.799999952316284, 3.799999952316284]
    composite_dataset = pd.DataFrame(
        {
            "a.text": a_text,
            "b.text": b_text,
            "similarity": similarity,
        },
    )
    _add_datatype(composite_dataset)

    assert (composite_dataset[DATA_TYPE_FIELD] == DatapointType.COMPOSITE).all()
    assert (composite_dataset["similarity"] == similarity).all()
    assert (composite_dataset["a"] == [{"text": text, DATA_TYPE_FIELD: DatapointType.TEXT} for text in a_text]).all()
    assert (composite_dataset["b"] == [{"text": text, DATA_TYPE_FIELD: DatapointType.TEXT} for text in b_text]).all()


def test__flatten_composite():
    composite_dataset = pd.DataFrame(
        {
            "a": [{"text": text, DATA_TYPE_FIELD: DatapointType.TEXT.value} for text in a_text],
            "b": [{"text": text, DATA_TYPE_FIELD: DatapointType.TEXT.value} for text in b_text],
            "c": [dict(text=a + b) for a, b in zip(a_text, b_text)],  # Should not flatten because no DATA_TYPE_FIELD
            DATA_TYPE_FIELD: DatapointType.COMPOSITE,
        },
    )
    composite_dataset = _flatten_composite(composite_dataset)

    expected = pd.DataFrame(
        {
            "a.text": a_text,
            f"a.{DATA_TYPE_FIELD}": DatapointType.TEXT,
            "b.text": b_text,
            f"b.{DATA_TYPE_FIELD}": DatapointType.TEXT,
            "c": [dict(text=a + b) for a, b in zip(a_text, b_text)],
            DATA_TYPE_FIELD: DatapointType.COMPOSITE,
        },
    )
    assert (composite_dataset.sort_index(axis=1) == expected.sort_index(axis=1)).all().all()


def test__add_datatype__invalid():
    composite_dataset = pd.DataFrame(
        {
            "a": [i for i in range(5)],
            "a.locator": [f"{i}.png" for i in range(5)],
        },
    )
    with pytest.raises(InputValidationError):
        _add_datatype(composite_dataset)

    composite_dataset = pd.DataFrame(
        {
            "too.many.dots": [i for i in range(5)],
        },
    )
    with pytest.raises(InputValidationError):
        _add_datatype(composite_dataset)

    composite_dataset = pd.DataFrame(
        {
            ".empty_prefix": [i for i in range(5)],
        },
    )
    with pytest.raises(InputValidationError):
        _add_datatype(composite_dataset)


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


def test__datapoint_dataframe__serde_composite() -> None:
    datapoints = [
        {
            "category": "outdoor" if i < 5 else "indoor",
            "is_same": True if i % 3 else False,
            "a.locator": f"https://test-iamge-{i}.png",
            "a.width": 500 + i,
            "a.height": 400 + i,
            "a.bboxes": [
                BoundingBox(top_left=(i, i), bottom_right=(i + 50, i + 50)) for i in range(random.randint(2, 6))
            ],
            "a.label": ClassificationLabel(label="dog"),
            "b.locator": f"https://test-iamge-{i}.png",
            "b.width": 500 + i,
            "b.height": 400 + i,
            "b.bboxes": [
                BoundingBox(top_left=(i, i), bottom_right=(i + 50, i + 50)) for i in range(random.randint(2, 6))
            ],
            "b.label": ClassificationLabel(label="cat"),
        }
        for i in range(10)
    ]
    df = pd.DataFrame(datapoints)
    df_expected = pd.DataFrame(
        dict(
            datapoint=[
                dict(
                    category=dp["category"],
                    is_same=dp["is_same"],
                    a={
                        DATA_TYPE_FIELD: DatapointType.IMAGE,
                        "label": dp["a.label"]._to_dict(),
                        "width": dp["a.width"],
                        "height": dp["a.height"],
                        "locator": dp["a.locator"],
                        "bboxes": [bbox._to_dict() for bbox in dp["a.bboxes"]],
                    },
                    b={
                        DATA_TYPE_FIELD: DatapointType.IMAGE,
                        "label": dp["b.label"]._to_dict(),
                        "width": dp["b.width"],
                        "height": dp["b.height"],
                        "locator": dp["b.locator"],
                        "bboxes": [bbox._to_dict() for bbox in dp["b.bboxes"]],
                    },
                    data_type=DatapointType.COMPOSITE,
                )
                for dp in datapoints
            ],
        ),
    )
    df_serialized = _to_serialized_dataframe(df, column=COL_DATAPOINT)
    df_deserialized = _to_deserialized_dataframe(df_serialized, column=COL_DATAPOINT)

    assert df_serialized[COL_DATAPOINT].apply(json.loads).equals(df_expected[COL_DATAPOINT])
    assert_frame_equal(df_deserialized.sort_index(axis=1), df.sort_index(axis=1))


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


def test__dataframe__serde_none__composite() -> None:
    column_name = COL_RESULT
    data = [
        ['{"a.city": "London"}'],
        ['{"a.city": "Tokyo"}'],
        [None],
    ]
    df_serialized = pd.DataFrame(data, columns=[column_name])

    df_expected = pd.DataFrame([["London"], ["Tokyo"], [np.nan]], columns=["a.city"])
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


def test__infer_id_fields() -> None:
    assert _infer_id_fields(
        pd.DataFrame(
            dict(
                locator=["s3://test.pdf", "https://test.png", "/home/test.mp4", "/tmp/test.pcd"],
            ),
        ),
    ) == ["locator"]
    assert _infer_id_fields(
        pd.DataFrame(
            dict(
                locator=["s3://test.pdf", "https://test.png", "/home/test.mp4", "/tmp/test.pcd"],
                text=["a", "b", "c", "d"],
            ),
        ),
    ) == ["locator"]
    assert _infer_id_fields(
        pd.DataFrame(
            dict(
                text=["a", "b", "c", "d"],
            ),
        ),
    ) == ["text"]
    assert _infer_id_fields(pd.DataFrame({"a.text": ["a", "b"], "b.text": ["c", "d"]})) == [
        "a.text",
        "b.text",
    ]

    try:
        assert _infer_id_fields(
            pd.DataFrame(
                dict(
                    text1=["a", "b", "c", "d"],
                ),
            ),
        )
    except Exception as e:
        assert str(e) == "Failed to infer the id_fields, please provide id_fields explicitly"
