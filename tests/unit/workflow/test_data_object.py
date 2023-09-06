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

from kolena.workflow._datatypes import DataObject


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

    tester = DataclassesTester(z=False, a="foobar", b=0.3, y=["hello"], x="world")
    serialized = tester._to_dict()
    assert list(serialized.keys()) == ["b", "a", "z", "y", "x"]

    # pydantic dataclass with `extra = allow` should have additional fields
    deserialized = DataclassesTester._from_dict(serialized)
    assert deserialized == tester
    assert deserialized.x == "world"
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
