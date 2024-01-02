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
import pytest

from kolena._utils.frozen import Frozen
from kolena.errors import FrozenObjectError


class TestFrozen(Frozen):
    def __init__(self) -> None:
        self.a = "a"
        self._freeze()


def test__frozen() -> None:
    obj = TestFrozen()
    with pytest.raises(FrozenObjectError):
        obj.a = "b"

    assert obj.a == "a"


def test__unfrozen() -> None:
    obj = TestFrozen()
    with obj._unfrozen():
        obj.a = "new"

    assert obj.a == "new"

    with pytest.raises(FrozenObjectError):
        obj.a = "b"
