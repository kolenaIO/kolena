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
import dataclasses
import sys

import pydantic
import pytest
from pydantic import Extra
from pydantic.dataclasses import dataclass

from kolena.workflow import BaseVideo
from kolena.workflow import Composite
from kolena.workflow import Document
from kolena.workflow import Image
from kolena.workflow import PointCloud
from kolena.workflow import Text
from kolena.workflow._datatypes import DataObject
from kolena.workflow.annotation import BitmapMask
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import BoundingBox3D
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import Keypoints
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import Polygon
from kolena.workflow.annotation import Polyline
from kolena.workflow.annotation import SegmentationMask
from kolena.workflow.asset import BaseVideoAsset
from kolena.workflow.asset import BinaryAsset
from kolena.workflow.asset import ImageAsset
from kolena.workflow.asset import PlainTextAsset
from kolena.workflow.asset import PointCloudAsset


# extensions must be frozen
@pytest.mark.skipif(sys.version_info < (3, 8), reason="Constraint not enforced on Python < 3.8")
def test__data_object__frozen() -> None:
    with pytest.raises(TypeError):

        @dataclasses.dataclass
        class DataclassesTester(DataObject):
            ...

    with pytest.raises(TypeError):

        @pydantic.dataclasses.dataclass
        class PydanticTester(DataObject):
            ...


# can use either stdlib dataclasses or pydantic dataclasses interchangeably
def test__data_object__dataclasses_or_pydantic() -> None:
    @dataclasses.dataclass(frozen=True)
    class DataclassesTester(DataObject):
        ...

    DataclassesTester()

    @pydantic.dataclasses.dataclass(frozen=True)
    class PydanticTester(DataObject):
        ...

    PydanticTester()


def test__data_object__serialize_order() -> None:
    @dataclasses.dataclass(frozen=True)
    class DataclassesTester(DataObject):
        b: float
        a: str
        z: bool

    tester = DataclassesTester(z=False, a="foobar", b=0.3)
    serialized = tester._to_dict()
    assert list(serialized.keys()) == ["b", "a", "z"]


def test__data_object__extras_allow() -> None:
    @dataclass(frozen=True, config={"extra": "allow"})
    class DataclassesTester(DataObject):
        b: float
        a: str
        z: bool

    # bbox = BoundingBox(top_left=(1, 1), bottom_right=(10, 10))
    bbox = LabeledBoundingBox(top_left=(1, 1), bottom_right=(10, 10), label="bus")
    tester = DataclassesTester(z=False, a="foobar", b=0.3, y=["hello"], x=bbox)
    serialized = tester._to_dict()
    assert list(serialized.keys()) == ["b", "a", "z", "y", "x"]

    # pydantic dataclass with `extra = allow` should have additional fields
    deserialized = DataclassesTester._from_dict(serialized)
    assert deserialized == tester
    assert deserialized.x == BoundingBox(top_left=bbox.top_left, bottom_right=bbox.bottom_right)
    assert deserialized.y == ["hello"]


def test__data_object__extras_allow_invalid() -> None:
    class Config:
        extra = Extra.allow

    @dataclass(frozen=True, config=Config)
    class DataclassesTester(DataObject):
        b: float
        a: str
        z: bool

    @dataclass(frozen=True)
    class CustomData:
        foo: str

    # extras should still be checked
    with pytest.raises(ValueError):
        DataclassesTester(z=False, a="foobar", b=0.3, y=CustomData(foo="bar"))._to_dict()


def test__data_object__extras_stdlib() -> None:
    @dataclasses.dataclass(frozen=True)
    class StdlibDataclassesTester(DataObject):
        a: str
        b: float
        z: bool

    serialized = dict(z=False, a="foobar", b=0.3, y=["hello"], x="world")

    # stdlib dataclass should still work
    stdlib_tester = StdlibDataclassesTester(z=False, a="foobar", b=0.3)
    stdlib_deserialized = StdlibDataclassesTester._from_dict(serialized)
    assert stdlib_deserialized == stdlib_tester


def test__data_object__extras_strict() -> None:
    @dataclass(frozen=True)
    class StrictDataclassesTester(DataObject):
        b: float
        a: str
        z: bool

    serialized = dict(z=False, a="foobar", b=0.3, y=["hello"], x="world")

    # pydantic dataclass without `extra = allow` should still work
    strict_tester = StrictDataclassesTester(z=False, a="foobar", b=0.3)
    strict_deserialized = StrictDataclassesTester._from_dict(serialized)
    assert strict_deserialized == strict_tester


def test__data_object__extras_ignore() -> None:
    @dataclass(frozen=True, config={"extra": "ignore"})
    class IgnoreExtraTester(DataObject):
        b: float
        a: str
        z: bool

    serialized = dict(z=False, a="foobar", b=0.3, y=["hello"], x="world")

    # pydantic dataclass with `extra = ignore` should still work
    tester = IgnoreExtraTester(z=False, a="foobar", b=0.3)
    deserialized = IgnoreExtraTester._from_dict(serialized)
    assert deserialized == tester


#
@pytest.mark.parametrize(
    "data_object",
    [
        BoundingBox(top_left=(1, 1), bottom_right=(11, 11), extra="foo"),
        Polygon(points=[(1, 1), (2, 2), (3, 3)], extra="foo"),
        Keypoints(points=[(i, i) for i in range(3)], extra="foo"),
        Polyline(points=[(i, i) for i in range(3)], extra="foo"),
        BoundingBox3D(center=(1, 1, 1), dimensions=(5, 5, 5), rotations=(0.1, 0.2, 0.3), extra="foo"),
        SegmentationMask(labels={1: "cat", 2: "dog"}, locator="s3://bbb.jpg", extra="foo"),
        BitmapMask(locator="s3://test.png", extra="foo"),
        ClassificationLabel(label="bus", extra="foo"),
        ImageAsset(locator="s3://test.jpg", extra="foo"),
        PlainTextAsset(locator="http://test.csv", extra="foo"),
        BinaryAsset(locator="http://test.bin", extra="foo"),
        PointCloudAsset(locator="https://test.pcd", extra="foo"),
        BaseVideoAsset(locator="https://test.mp4", extra="foo"),
        Composite(extra="foo"),
        Image(locator="s3//img1.png", extra="foo"),
        Text(text="foo bar", extra="foo"),
        BaseVideo(locator="https://test.mp4", extra="foo"),
        Document(locator="s3://test.pdf", extra="foo"),
        PointCloud(locator="s3://test.pcd", extra="foo"),
    ],
)
def test__data_object__extras_deserialize_builtins(data_object) -> None:
    @dataclass(frozen=True, config={"extra": "allow"})
    class Tester(DataObject):
        ...

    tester = Tester(a=data_object, b=[data_object, data_object])
    serialized = tester._to_dict()
    deserialized = Tester._from_dict(serialized)
    assert deserialized == tester
    assert deserialized.a == data_object
    assert deserialized.a.extra == "foo"
    assert deserialized.b == [data_object, data_object]
    assert deserialized.b[1].extra == "foo"
