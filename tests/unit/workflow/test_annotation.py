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
from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Tuple

from kolena.workflow._datatypes import DATA_TYPE_FIELD
from kolena.workflow._datatypes import DataObject
from kolena.workflow.annotation import _AnnotationType
from kolena.workflow.annotation import BitmapMask
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import BoundingBox3D
from kolena.workflow.annotation import Keypoints
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledBoundingBox3D
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import Polygon
from kolena.workflow.annotation import Polyline
from kolena.workflow.annotation import SegmentationMask


def test__serialize__simple() -> None:
    assert BoundingBox(top_left=(1, 1), bottom_right=(2, 2))._to_dict() == {
        "top_left": [1, 1],
        "bottom_right": [2, 2],
        DATA_TYPE_FIELD: f"{_AnnotationType._data_category()}/{_AnnotationType.BOUNDING_BOX.value}",
    }


def test__serialize__nested() -> None:
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
    )
    obj_dict = obj._to_dict()

    assert obj_dict["b"] == {
        "top_left": [0, 0],
        "bottom_right": [1, 1],
        DATA_TYPE_FIELD: f"{_AnnotationType._data_category()}/{_AnnotationType.BOUNDING_BOX.value}",
    }
    assert Tester._from_dict(obj_dict) == obj
