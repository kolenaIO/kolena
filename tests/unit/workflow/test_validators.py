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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pytest

from kolena.workflow._datatypes import DataObject
from kolena.workflow._validators import validate_data_object_type
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import Polygon
from kolena.workflow.annotation import Polyline
from kolena.workflow.asset import ImageAsset


def test__validate_data_object__invalid() -> None:
    @dataclasses.dataclass(frozen=True)
    class NonDataObject:
        ...

    with pytest.raises(ValueError):
        validate_data_object_type(NonDataObject)  # type: ignore


def test__validate_field__scalar() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(DataObject):
        a: str
        b: bool
        c: int
        d: float

    validate_data_object_type(Tester)


def test__validate_field__annotation() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(DataObject):
        a: BoundingBox
        b: LabeledBoundingBox
        c: Polygon
        d: LabeledPolygon
        e: Keypoints
        f: Polyline

    validate_data_object_type(Tester)


def test__validate_field__asset() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(DataObject):
        a: ImageAsset

    validate_data_object_type(Tester)


def test__validate_field__list() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(DataObject):
        a: List[str]
        b: List[bool]
        c: List[int]
        d: List[float]
        e: List[BoundingBox]
        f: List[LabeledBoundingBox]
        g: List[Polygon]
        h: List[LabeledPolygon]
        i: List[Keypoints]
        j: List[Polyline]
        k: List[Union[BoundingBox, LabeledBoundingBox]]

    validate_data_object_type(Tester)


def test__validate_field__list__invalid() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(DataObject):
        a: List[Dict[str, Any]]

    with pytest.raises(ValueError):
        validate_data_object_type(Tester)


def test__validate_field__list_of_list__invalid() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(DataObject):
        a: List[List[str]]

    with pytest.raises(ValueError):
        validate_data_object_type(Tester)


def test__validate_field__optional() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(DataObject):
        a: Optional[str]
        b: Optional[bool]
        c: Optional[int]
        d: Optional[float]
        e: Optional[ImageAsset]

    validate_data_object_type(Tester)


def test__validate_field__optional__invalid() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(DataObject):
        a: Optional[Dict[str, Any]]

    with pytest.raises(ValueError):
        validate_data_object_type(Tester)


def test__validate_field__union() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(DataObject):
        a: Union[str, bool, ImageAsset, List[Union[BoundingBox, Polygon]]]

    validate_data_object_type(Tester)


def test__validate_field__union__invalid() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(DataObject):
        a: Union[str, float, int, Dict[str, Any]]

    with pytest.raises(ValueError):
        validate_data_object_type(Tester)


def test__validate_field__invalid() -> None:
    @dataclasses.dataclass(frozen=True)
    class BytesTester(DataObject):
        a: bytes

    with pytest.raises(ValueError):
        validate_data_object_type(BytesTester)

    @dataclasses.dataclass(frozen=True)
    class DictTester(DataObject):
        a: Dict[str, Any]

    with pytest.raises(ValueError):
        validate_data_object_type(DictTester)

    @dataclasses.dataclass(frozen=True)
    class Nested(DataObject):
        a: int

    @dataclasses.dataclass(frozen=True)
    class NestedTester(DataObject):
        nested: Nested

    with pytest.raises(ValueError):
        validate_data_object_type(NestedTester)
