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
from contextlib import contextmanager
from typing import Any
from typing import Dict
from typing import Iterator
from typing import TypeVar

from kolena.errors import FrozenObjectError

T = TypeVar("T")


class Frozen:
    _frozen = False

    def _freeze(self) -> None:
        object.__setattr__(self, "_frozen", True)

    @contextmanager
    def _unfrozen(self) -> Iterator[None]:
        object.__setattr__(self, "_frozen", False)
        yield
        self._freeze()

    def _fields(self) -> Dict[str, Any]:
        return {key: value for key, value in self.__dict__.items() if not key.startswith("_")}

    def __setattr__(self, key: str, value: T) -> None:
        if self._frozen:
            raise FrozenObjectError("cannot set attribute of frozen object")
        object.__setattr__(self, key, value)

    def __repr__(self) -> str:
        fields = ", ".join(f"{key}={repr(value)}" for key, value in self._fields().items())
        return f"{type(self).__name__}({fields})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and self.__dict__ == other.__dict__
