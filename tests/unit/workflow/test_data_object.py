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

from kolena.workflow._datatypes import DataObject


# extensions must be frozen
@pytest.mark.skipif(sys.version_info < (3, 8), reason="Constraint not enforced on Python < 3.8")
def test_frozen() -> None:
    with pytest.raises(TypeError):

        @dataclasses.dataclass
        class DataclassesTester(DataObject):
            ...

    with pytest.raises(TypeError):

        @pydantic.dataclasses.dataclass
        class PydanticTester(DataObject):
            ...


# can use either stdlib dataclasses or pydantic dataclasses interchangeably
def test_dataclasses_or_pydantic() -> None:
    @dataclasses.dataclass(frozen=True)
    class DataclassesTester(DataObject):
        ...

    DataclassesTester()

    @pydantic.dataclasses.dataclass(frozen=True)
    class PydanticTester(DataObject):
        ...

    PydanticTester()


def test_dataclasses_serialize_order() -> None:
    @dataclasses.dataclass(frozen=True)
    class DataclassesTester(DataObject):
        b: float
        a: str
        z: bool

    tester = DataclassesTester(z=False, a="foobar", b=0.3)
    serialized = tester._to_dict()
    assert list(serialized.keys()) == ["b", "a", "z"]
