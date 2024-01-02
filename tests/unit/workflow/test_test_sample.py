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
import math
from typing import Any
from typing import Callable
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import dacite
import numpy as np
import pydantic
import pytest

from kolena._utils.serde import as_deserialized_json
from kolena._utils.serde import as_serialized_json
from kolena.workflow import Audio
from kolena.workflow import BaseVideo
from kolena.workflow import Composite
from kolena.workflow import Document
from kolena.workflow import Image
from kolena.workflow import ImagePair
from kolena.workflow import ImageText
from kolena.workflow import Metadata
from kolena.workflow import PointCloud
from kolena.workflow import TestSample
from kolena.workflow import Text
from kolena.workflow import Video
from kolena.workflow._datatypes import DataObject
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import Polygon
from kolena.workflow.annotation import Polyline
from kolena.workflow.asset import ImageAsset
from kolena.workflow.test_sample import _TestSampleType
from kolena.workflow.test_sample import _validate_test_sample_type


@pytest.mark.parametrize("base_type", [Image, ImageText, Text, BaseVideo])
def test__validate(base_type: Type[TestSample]) -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(base_type):
        a: int
        b: float
        c: Optional[bool]
        d: Union[str, int]
        e: List[float]
        f: BoundingBox
        g: Polygon
        metadata: Dict[str, List[float]]

    _validate_test_sample_type(Tester)


@pytest.mark.parametrize("base_type", [Video])
def test__validate__default_fields(base_type: Type[TestSample]) -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(base_type):
        a: Optional[bool] = True
        b: Optional[BoundingBox] = None
        c: List[float] = dataclasses.field(default_factory=list)

    _validate_test_sample_type(Tester)

    @pydantic.dataclasses.dataclass(frozen=True)
    class Tester(base_type):
        a: Optional[bool] = True
        b: Optional[BoundingBox] = None
        c: List[float] = dataclasses.field(default_factory=list)

    _validate_test_sample_type(Tester)


def test__validate__metadata() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        metadata: Metadata

    _validate_test_sample_type(Tester)


@pytest.mark.parametrize(
    "value_type",
    [
        str,
        int,
        float,
        bool,
        Union[str, int],
        Optional[bool],
        List[float],
        Union[str, List[str]],
        Optional[Union[int, float]],
        None,
        Union[str, None],
    ],
)
def test__validate__custom_metadata(value_type: Type) -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        metadata: Dict[str, value_type]

    _validate_test_sample_type(Tester)


@pytest.mark.parametrize(
    "key_type",
    [
        int,
        float,
        bool,
        Optional[str],
        Union[str, int],
        List[str],
        BoundingBox,
        Polygon,
        ImageAsset,
    ],
)
def test__validate__invalid_metadata_key(key_type: Type) -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        metadata: Dict[key_type, str]

    with pytest.raises(ValueError):
        _validate_test_sample_type(Tester)


@pytest.mark.parametrize(
    "value_type",
    [
        BoundingBox,
        Polygon,
        ImageAsset,
        Dict[str, str],
        Union[str, Polyline],
        Optional[Any],
        Any,
    ],
)
def test__validate__invalid_metadata_value(value_type: Type) -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        metadata: Dict[str, value_type]

    with pytest.raises(ValueError):
        _validate_test_sample_type(Tester)


@pytest.mark.parametrize("base_type", [Image, ImageText, Text, BaseVideo])
def test__validate__invalid(base_type: Type[TestSample]) -> None:
    @dataclasses.dataclass(frozen=True)
    class BytesTester(base_type):
        a: bytes

    with pytest.raises(ValueError):
        _validate_test_sample_type(BytesTester)

    @dataclasses.dataclass(frozen=True)
    class DictTester(base_type):
        a: Dict[str, Any]

    with pytest.raises(ValueError):
        _validate_test_sample_type(DictTester)

    @dataclasses.dataclass(frozen=True)
    class Nested(DataObject):
        a: float

    @dataclasses.dataclass(frozen=True)
    class NestedTester(base_type):
        a: Nested

    with pytest.raises(ValueError):
        _validate_test_sample_type(NestedTester)


def test__validate__composite() -> None:
    @dataclasses.dataclass(frozen=True)
    class Nested(Image):
        a: float
        b: BoundingBox

    @dataclasses.dataclass(frozen=True)
    class NestedTester(ImagePair):
        a: Nested
        b: Nested
        c: float
        d: BoundingBox
        e: Optional[ImageAsset]

    _validate_test_sample_type(NestedTester)


def test__validate__composite__invalid() -> None:
    @dataclasses.dataclass(frozen=True)
    class Nested(DataObject):
        a: float

    @dataclasses.dataclass(frozen=True)
    class NestedTester(TestSample):  # must extend CompositeTestSample
        a: Nested
        b: Nested

    with pytest.raises(ValueError):
        _validate_test_sample_type(NestedTester)

    @dataclasses.dataclass(frozen=True)
    class InnerInner(TestSample):
        a: float

    @dataclasses.dataclass(frozen=True)
    class Inner(Image):
        a: InnerInner

    @dataclasses.dataclass(frozen=True)
    class DoubleNestedTester(ImagePair):
        a: Inner

    with pytest.raises(ValueError):
        _validate_test_sample_type(DoubleNestedTester)


def test__validate__composite_invalid_nested() -> None:
    @dataclasses.dataclass(frozen=True)
    class Nested(Composite):
        a: float
        b: Image

    @dataclasses.dataclass(frozen=True)
    class NestedTester(Composite):
        a: Nested
        b: Nested

    with pytest.raises(ValueError, match="Nested composite"):
        _validate_test_sample_type(NestedTester)


def test__validate__custom() -> None:
    @dataclasses.dataclass(frozen=True)
    class CustomTester(TestSample):
        a: int
        # self.data_type should be CUSTOM by default

    _validate_test_sample_type(CustomTester)


def test__validate__custom__invalid() -> None:
    @dataclasses.dataclass(frozen=True)
    class CustomTester(TestSample):
        a: int

        @classmethod
        def _data_type(cls) -> _TestSampleType:
            return _TestSampleType.VIDEO  # must be CUSTOM if extending TestSample and not a builtin base type

    with pytest.raises(ValueError):
        _validate_test_sample_type(CustomTester)


def test__validate__invalid_generic() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: Sequence[int]  # typing.Sequence is not supported

    with pytest.raises(ValueError):
        _validate_test_sample_type(Tester)


def test__serialize() -> None:
    @pydantic.dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: str
        b: bool
        c: int
        d: float
        e: BoundingBox
        f: Polyline
        g: ImageAsset
        h: List[int]
        i: List[Polygon]
        j: Tuple[int, str, bool]
        k: Optional[str]
        l: Optional[LabeledPolygon]
        m: Union[float, bool, BoundingBox]
        n: Union[List[int], List[Keypoints]]

    obj = Tester(
        a="abc",
        b=True,
        c=123,
        d=0.1,
        e=BoundingBox(top_left=(0, 0), bottom_right=(10, 10)),
        f=Polyline(points=[(1, 1), (2, 2)]),
        g=ImageAsset(locator="s3://test/locator.png"),
        h=[1, 2, 3],
        i=[Polygon(points=[(0, 0), (1, 1), (2, 2)])],
        j=(1, "test", False),
        k=None,
        l=LabeledPolygon(label="test", points=[(1, 1), (2, 2), (3, 3)]),
        m=BoundingBox(top_left=(0, 0), bottom_right=(10, 10)),
        n=[1, 2, 3],
    )

    obj_dict = obj._to_dict()
    assert obj_dict == dict(
        a="abc",
        b=True,
        c=123,
        d=0.1,
        e=BoundingBox(top_left=(0, 0), bottom_right=(10, 10))._to_dict(),
        f=Polyline(points=[(1, 1), (2, 2)])._to_dict(),
        g=ImageAsset(locator="s3://test/locator.png")._to_dict(),
        h=[1, 2, 3],
        i=[Polygon(points=[(0, 0), (1, 1), (2, 2)])._to_dict()],
        j=[1, "test", False],  # cast to list as tuples are not JSON
        k=None,
        l=LabeledPolygon(label="test", points=[(1, 1), (2, 2), (3, 3)])._to_dict(),
        m=BoundingBox(top_left=(0, 0), bottom_right=(10, 10))._to_dict(),
        n=[1, 2, 3],
        data_type=f"{_TestSampleType._data_category()}/{_TestSampleType.CUSTOM.value}",
    )

    assert Tester._from_dict(obj_dict) == obj


def test__serialize__structured() -> None:
    @pydantic.dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: List[str]
        b: Tuple[bool, int, bool]
        c: Union[int, float]
        d: Union[float, int]
        e: Optional[Union[float, int]]
        f: Optional[Union[float, int]]
        g: List[Optional[Union[float, int]]]
        h: List[List[List[Optional[int]]]]
        i: Optional[int] = None

    obj = Tester(
        a=["a", "b", "c"],
        b=(False, 0, True),
        c=1.23,
        d=2,
        e=None,
        f=5,
        g=[None, 1, 1.1],
        h=[[], [[]], [[None], []], [[1, 2]], [[None, 1], [], [1, None]]],
        # note omission of i
    )

    assert Tester._from_dict(obj._to_dict()) == obj


def test__serialize__missing_required() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: int

    with pytest.raises(ValueError):
        Tester._from_dict({})


def test__serialize__default() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: int = dataclasses.field(default=123)

    assert Tester._from_dict({}) == Tester()


def test__serialize__default_factory() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: Keypoints = dataclasses.field(default_factory=lambda: Keypoints(points=[]))
        b: List[bool] = dataclasses.field(default_factory=lambda: [True, False])
        c: List[Union[None, str]] = dataclasses.field(default_factory=list)

    assert Tester._from_dict({}) == Tester()
    assert Tester._from_dict(dict(c=[None, "test"])) == Tester(c=[None, "test"])


def test__serialize__structured__optional() -> None:
    @pydantic.dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: List[Optional[BoundingBox]]

    obj = Tester(
        a=[
            BoundingBox(top_left=(0, 0), bottom_right=(10, 10)),
            None,
            BoundingBox(top_left=(10, 10), bottom_right=(20, 20)),
        ],
    )

    assert Tester._from_dict(obj._to_dict()) == obj


def test__serialize__structured__union() -> None:
    @pydantic.dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        # TODO: ordering matters here -- if BoundingBox is declared before LabeledBoundingBox, the object will
        #  deserialize as a BoundingBox and drop the label. Not ideal, but at least manageable
        a: List[Union[LabeledBoundingBox, BoundingBox, Polygon]]

    obj = Tester(
        a=[
            BoundingBox(top_left=(0, 0), bottom_right=(10, 10)),
            LabeledBoundingBox(label="test", top_left=(10, 10), bottom_right=(20, 20)),
            Polygon(points=[(0, 0), (10, 0), (10, 10), (0, 10)]),
        ],
    )

    assert Tester._from_dict(obj._to_dict()) == obj


def test__serialize__cast() -> None:
    @pydantic.dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: str
        b: str

    with pytest.raises(ValueError):
        Tester._from_dict(dict(a="has"))


def test__serialize__missing_keys() -> None:
    @pydantic.dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: List[str]
        b: float
        c: bool
        d: List[Tuple[float, float]]

    obj = Tester._from_dict(dict(a=[1, 2, 3], b=1, c=1, d=[[1, 1]]))
    assert obj.a == ["1", "2", "3"]
    assert obj.b == 1.0
    assert obj.c is True
    assert obj.d == [(1.0, 1.0)]


def test__serialize__extra_keys() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: str

    obj = Tester._from_dict(dict(a="has", b="extra", c="fields"))  # does not throw
    assert obj.a == "has"
    # TODO: make extra available somehow?


@pytest.mark.parametrize("dataclass_decorator", [dataclasses.dataclass, pydantic.dataclasses.dataclass])
def test__serialize__numpy__scalar(dataclass_decorator: Callable[..., Any]) -> None:
    @dataclass_decorator(frozen=True)
    class Tester(TestSample):
        a: bool
        b: int
        c: float

    obj = Tester(a=np.bool_(True), b=np.int32(1), c=np.float64(0.5))
    assert Tester._from_dict(obj._to_dict()) == Tester(a=True, b=1, c=0.5)


@pytest.mark.parametrize(
    "dataclass_decorator",
    [
        dataclasses.dataclass,
        # NOTE: pydantic blocks np.array in place of list, which is fine, as it's technically correct -- we don't need
        #  to support that case if users choose to use pydantic dataclasses
        # pydantic.dataclasses.dataclass,
    ],
)
def test__serialize__numpy__array(dataclass_decorator: Callable[..., Any]) -> None:
    @dataclass_decorator(frozen=True)
    class Tester1D(TestSample):
        a: List[int]

    obj = Tester1D(a=np.array([1, 2, 3]))
    assert Tester1D._from_dict(obj._to_dict()) == Tester1D(a=[1, 2, 3])

    @dataclass_decorator(frozen=True)
    class Tester2D(TestSample):
        a: List[List[int]]

    obj = Tester2D(a=np.array([[1, 2], [3, 4]]))
    assert Tester2D._from_dict(obj._to_dict()) == Tester2D(a=[[1, 2], [3, 4]])


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, -np.nan, float("NaN"), float("inf"), -float("inf")])
def test__serialize__nan(value: float) -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: float

    obj = Tester(value)
    serialized = as_serialized_json(obj._to_dict())
    deserialized = as_deserialized_json(serialized)
    deserialized_obj = dacite.from_dict(Tester, data=deserialized)
    if math.isnan(value):
        assert math.isnan(deserialized_obj.a)
    else:
        assert deserialized_obj.a == value


@pytest.mark.parametrize("dataclass", [dataclasses.dataclass, pydantic.dataclasses.dataclass])
def test__serialize__metadata(dataclass: Callable) -> None:
    @dataclass(frozen=True)
    class Tester(TestSample):
        a: float
        metadata: Metadata

    metadata_dict = dict(b=2.0, c=True, d="d", e=None)
    obj = Tester(a=1, metadata=metadata_dict)
    got_dict = obj._to_dict()
    got_metadata_dict = obj._to_metadata_dict()
    assert len(got_dict) == 2
    assert got_dict["a"] == 1
    assert got_dict["data_type"] == "TEST_SAMPLE/CUSTOM"
    assert got_metadata_dict == metadata_dict


def test__serialize__metadata__absent() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: float

    obj = Tester(a=1)
    got_dict = obj._to_dict()
    got_metadata_dict = obj._to_metadata_dict()
    assert len(got_dict) == 2
    assert got_dict["a"] == 1
    assert got_dict["data_type"] == "TEST_SAMPLE/CUSTOM"
    assert got_metadata_dict == {}


# TODO: ideally we wouldn't require a default_factory for metadata, as metadata isn't loaded during test running. Not
#  loading metadata during testing effectively requires metadata always be declared with a default
@pytest.mark.skip("not implemented")
def test__deserialize__metadata__absent_no_default() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(TestSample):
        a: int
        metadata: Metadata

    assert Tester._from_dict(dict(a=1)) == Tester(a=1, metadata={})


@pytest.mark.parametrize(
    "dataclass,include_default",
    [
        (dataclasses.dataclass, True),
        (pydantic.dataclasses.dataclass, True),
        (dataclasses.dataclass, False),
        (pydantic.dataclasses.dataclass, False),
    ],
)
def test__deserialize__metadata(dataclass: Callable, include_default: bool) -> None:
    @dataclass(frozen=True)
    class Tester(TestSample):
        metadata: Metadata = dataclasses.field(**(dict(default_factory=dict) if include_default else dict()))

    metadata_dict = dict(a=1, b=2.0, c=True, d="d", e=None, f=[1, 2.0, True, "d", None])
    obj = Tester(metadata=metadata_dict)
    got_dict = obj._to_dict()
    got_metadata_dict = obj._to_metadata_dict()
    assert obj == Tester._from_dict(dict(**got_dict, metadata=got_metadata_dict))


@pytest.mark.parametrize("dataclass", [dataclasses.dataclass, pydantic.dataclasses.dataclass])
def test__instantiate__video(dataclass: Callable) -> None:
    @dataclass(frozen=True)
    class Tester(Video):
        a: int = 0

    locator = "s3://test-bucket/video.mp4"
    thumbnail = ImageAsset(locator="https://example.com/test.png")
    Tester(locator=locator, thumbnail=thumbnail, start=0, end=1, a=1)
    Tester(locator=locator)
    Tester(locator=locator, start=1, end=1.5)
    Tester(locator=locator, a=1)

    with pytest.raises(ValueError):
        Tester(locator=locator, start=1, end=0)

    with pytest.raises(ValueError):
        Tester(locator=locator, start=-1, end=0)

    with pytest.raises(ValueError):
        Tester(locator=locator, start=-2, end=-1)


@pytest.mark.parametrize("dataclass", [dataclasses.dataclass, pydantic.dataclasses.dataclass])
def test__instantiate__document(dataclass: Callable) -> None:
    @dataclass(frozen=True)
    class Tester(Document):
        example: Optional[str] = None

    locator = "s3://bucket/path/to/document.pdf"
    Tester(locator=locator, example="test")
    Tester(locator=locator, example=None)
    Tester(locator=locator)


@pytest.mark.parametrize("dataclass", [dataclasses.dataclass, pydantic.dataclasses.dataclass])
def test__instantiate__pointcloud(dataclass: Callable) -> None:
    @dataclass(frozen=True)
    class Tester(PointCloud):
        width: int
        height: int

    locator = "s3://bucket/path/to/pointcloud.pcd"
    Tester(locator=locator, width=10, height=10)


@pytest.mark.parametrize("dataclass", [dataclasses.dataclass, pydantic.dataclasses.dataclass])
def test__instantiate__audio(dataclass: Callable) -> None:
    @dataclass(frozen=True)
    class Tester(Audio):
        length: int

    locator = "s3://bucket/path/to/audio.mp3"
    Tester(locator=locator, length=10)
