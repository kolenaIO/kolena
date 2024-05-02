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
import dataclasses
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import pandas as pd
import pydantic
import pytest

from kolena._utils.datatypes import DATA_TYPE_FIELD
from kolena._utils.datatypes import DataObject
from kolena.annotation import _AnnotationType
from kolena.annotation import BitmapMask
from kolena.annotation import BoundingBox
from kolena.annotation import BoundingBox3D
from kolena.annotation import Keypoints
from kolena.annotation import LabeledBoundingBox
from kolena.annotation import LabeledBoundingBox3D
from kolena.annotation import LabeledPolygon
from kolena.annotation import LabeledTextSegments
from kolena.annotation import Polygon
from kolena.annotation import Polyline
from kolena.annotation import SegmentationMask
from kolena.annotation import TextSegment


def test__serde__simple() -> None:
    obj = LabeledPolygon(points=[(1, 1), (2, 2), (3, 3)], label="test")
    obj_dict = obj._to_dict()
    assert obj_dict == {
        "label": "test",
        "points": [[1, 1], [2, 2], [3, 3]],
        DATA_TYPE_FIELD: f"{_AnnotationType._data_category().value}/{_AnnotationType.POLYGON.value}",
    }
    assert LabeledPolygon._from_dict(obj_dict) == obj


def test__serde__derived() -> None:
    obj = BoundingBox(top_left=(0, 0), bottom_right=(0, 0))
    obj_dict = obj._to_dict()
    assert obj_dict == {
        "top_left": [0, 0],
        "bottom_right": [0, 0],
        "width": 0,
        "height": 0,
        "area": 0,
        "aspect_ratio": 0,
        DATA_TYPE_FIELD: f"{_AnnotationType._data_category().value}/{_AnnotationType.BOUNDING_BOX.value}",
    }
    # deserialization from dict containing all fields, including derived
    assert BoundingBox._from_dict(obj_dict) == obj
    # deserialization from dict containing only non-derived fields
    assert BoundingBox._from_dict({k: obj_dict[k] for k in ["top_left", "bottom_right", DATA_TYPE_FIELD]}) == obj


@pytest.mark.parametrize("dataclass_decorator", [dataclasses.dataclass, pydantic.dataclasses.dataclass])
def test__serde__derived__extended(dataclass_decorator: Callable[..., Any]) -> None:
    @dataclass_decorator(frozen=True)
    class ExtendedBoundingBox(BoundingBox):
        a: str
        b: bool = False
        c: Optional[int] = None

    obj = ExtendedBoundingBox(top_left=(0, 0), bottom_right=(0, 0), a="a", c=0)
    obj_dict = obj._to_dict()
    assert obj_dict == {
        "top_left": [0, 0],
        "bottom_right": [0, 0],
        "width": 0,
        "height": 0,
        "area": 0,
        "aspect_ratio": 0,
        "a": "a",
        "b": False,
        "c": 0,
        DATA_TYPE_FIELD: f"{_AnnotationType._data_category().value}/{_AnnotationType.BOUNDING_BOX.value}",
    }
    assert ExtendedBoundingBox._from_dict(obj_dict) == obj


def test__serde__nested() -> None:
    @dataclass(frozen=True)
    class Tester(DataObject):
        b: BoundingBox
        c: LabeledBoundingBox
        d: Polygon
        e: LabeledPolygon
        f: Polyline
        g: Keypoints
        h: Optional[BoundingBox]
        i: Optional[Polyline]
        j: Tuple[BoundingBox, Polygon]
        k: List[Polyline]
        l: BoundingBox3D
        m: Optional[LabeledBoundingBox3D]
        n: SegmentationMask
        o: BitmapMask
        p: LabeledTextSegments

    obj = Tester(
        b=BoundingBox(top_left=(0, 0), bottom_right=(1, 1)),
        c=LabeledBoundingBox(label="c", top_left=(10, 10), bottom_right=(100, 100)),
        d=Polygon(points=[(0, 0), (1, 1), (2, 2), (0, 0)]),
        e=LabeledPolygon(label="e", points=[(0, 0), (1, 1), (2, 2), (0, 0)]),
        f=Polyline(points=[(0, 0), (1, 1), (2, 2)]),
        g=Keypoints(points=[(10, 10), (11, 11), (12, 12)]),
        h=None,
        i=Polyline(points=[(0, 0), (1, 1)]),
        j=(BoundingBox(top_left=(5, 6), bottom_right=(7, 8)), Polygon(points=[(1, 1), (2, 2), (3, 3)])),
        k=[Polyline(points=[(1, 1), (2, 2)]), Polyline(points=[(3, 3), (4, 4)])],
        l=BoundingBox3D(center=(0, 1, 2), dimensions=(3, 4, 5), rotations=(0, 10, 0)),
        m=None,
        n=SegmentationMask(labels={1: "cat", 10: "dog"}, locator="s3://abc"),
        o=BitmapMask(locator="s3://def"),
        p=LabeledTextSegments(text_segments=[TextSegment(text_field="text", segments=[(3, 8), (9, 21)])], label="name"),
    )
    obj_dict = obj._to_dict()

    assert obj_dict["e"] == {
        "points": [[0, 0], [1, 1], [2, 2], [0, 0]],
        "label": "e",
        DATA_TYPE_FIELD: f"{_AnnotationType._data_category().value}/{_AnnotationType.POLYGON.value}",
    }
    assert Tester._from_dict(obj_dict) == obj


@pytest.mark.parametrize(
    "top_left,bottom_right,expected",
    [
        ((0, 0), (0, 0), dict(width=0, height=0, area=0, aspect_ratio=0)),
        ((10, 10), (10, 10), dict(width=0, height=0, area=0, aspect_ratio=0)),
        # test different permutations to ensure that we're robust to coordinate swapping
        ((10, 10), (20, 30), dict(width=10, height=20, area=200, aspect_ratio=0.5)),
        ((20, 30), (10, 10), dict(width=10, height=20, area=200, aspect_ratio=0.5)),
        ((20, 10), (10, 30), dict(width=10, height=20, area=200, aspect_ratio=0.5)),
        ((10, 30), (20, 10), dict(width=10, height=20, area=200, aspect_ratio=0.5)),
    ],
)
def test__bounding_box__derived(
    top_left: Tuple[float, float],
    bottom_right: Tuple[float, float],
    expected: Dict[str, float],
) -> None:
    bbox = BoundingBox(top_left=top_left, bottom_right=bottom_right)
    for field, expected_value in expected.items():
        assert getattr(bbox, field) == expected_value


@pytest.mark.parametrize(
    "dimensions,expected",
    [
        ((0, 0, 0), 0),
        ((10, 10, 10), 1000),
        ((1, 1, 1), 1),
        ((1000, 0, 1000), 0),
    ],
)
def test__bounding_box_3d__derived(dimensions: Tuple[float, float, float], expected: float) -> None:
    bbox = BoundingBox3D(center=(0, 0, 0), dimensions=dimensions, rotations=(0, 0, 0))
    assert bbox.volume == expected


@pytest.mark.parametrize(
    "text,keywords,expected_segments",
    [
        (
            "Kolena is a comprehensive machine learning testing and debugging platform to surface hidden"
            " model behaviors and take the mystery out of model development.",
            ["Kolena", "machine learning", "model development"],
            [(0, 6), (26, 42), (136, 153)],
        ),
        ("The king himself told the audience that he is him", ["queen", "him"], [(46, 49)]),
        (
            "Creating fine-grained tests is labor-intensive and typically involves manual annotation of countless "
            "images, a costly and time-consuming process",
            ["fine-grained tests", "manual annotation", "and"],
            [(9, 27), (70, 87), (47, 50), (118, 121)],
        ),
    ],
)
def test__extract_labeled_text_segments_from_keywords(
    text: str,
    keywords: list[str],
    expected_segments: list[tuple[int, int]],
) -> None:
    df = pd.DataFrame({"text": [text]})
    labeled_text_segments = LabeledTextSegments.extract_labeled_text_segments_from_keywords(
        df,
        ["text"],
        {"test_label": keywords},
        colors={"test_label": "red"},
    )
    actual_segments_row = labeled_text_segments["labeled_text_segments"].iloc[0]
    expected = [
        LabeledTextSegments(
            text_segments=[TextSegment(text_field="text", segments=expected_segments)],
            label="test_label",
            color="red",
        ),
    ]
    assert actual_segments_row == expected
    for start, end in actual_segments_row[0].text_segments[0].segments:
        assert text[start:end] in keywords


def test__extract_labeled_text_segments_from_keywords__multi_label_and_field() -> None:
    text1 = "Perform high-resolution model evaluation"
    text2 = "Understand and track behavioral improvements and regressions"
    text3 = "Meaningfully communicate model capabilities"
    text4 = "Automate model testing and deployment workflows"
    df = pd.DataFrame({"text_field1": [text1, text2], "text_field2": [text3, text4]})
    labeled_text_segments = LabeledTextSegments.extract_labeled_text_segments_from_keywords(
        df,
        ["text_field1", "text_field2"],
        {
            "test_label1": ["model", "track"],
            "test_label2": ["evaluation", "testing", "communicate"],
        },
        colors={
            "test_label1": "red",
            "test_label2": "green",
        },
    )
    actual_segments_row_1 = labeled_text_segments["labeled_text_segments"].iloc[0]
    actual_segments_row_2 = labeled_text_segments["labeled_text_segments"].iloc[1]
    expected_segments_row_1 = [
        LabeledTextSegments(
            text_segments=[
                TextSegment(text_field="text_field1", segments=[(24, 29)]),
                TextSegment(text_field="text_field2", segments=[(25, 30)]),
            ],
            label="test_label1",
            color="red",
        ),
        LabeledTextSegments(
            text_segments=[
                TextSegment(text_field="text_field1", segments=[(30, 40)]),
                TextSegment(text_field="text_field2", segments=[(13, 24)]),
            ],
            label="test_label2",
            color="green",
        ),
    ]

    expected_segments_row_2 = [
        LabeledTextSegments(
            text_segments=[
                TextSegment(text_field="text_field1", segments=[(15, 20)]),
                TextSegment(text_field="text_field2", segments=[(9, 14)]),
            ],
            label="test_label1",
            color="red",
        ),
        LabeledTextSegments(
            text_segments=[TextSegment(text_field="text_field2", segments=[(15, 22)])],
            label="test_label2",
            color="green",
        ),
    ]

    assert actual_segments_row_1 == expected_segments_row_1
    assert actual_segments_row_2 == expected_segments_row_2
