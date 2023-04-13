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
import typing
from typing import Any
from typing import Generic
from typing import Type
from typing import TypeVar

from kolena.errors import DirectInstantiationError
from kolena.errors import IncorrectUsageError

T = TypeVar("T")
U = TypeVar("U", bound="Uninstantiable")


class Uninstantiable(Generic[T]):
    __slots__ = ("data",)
    __frozen__ = False

    data: T

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise DirectInstantiationError(
            f"{type(self).__name__} is not directly instantiatable, "
            "please use one of the available static constructors",
        )

    @typing.no_type_check
    def __setattr__(self, key, value):
        if self.__frozen__:
            raise IncorrectUsageError("cannot modify frozen class")
        object.__setattr__(self, key, value)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(data={repr(self.data)})"

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, type(self)) and other.data == self.data

    @classmethod
    def __factory__(cls: Type[U], data: Any) -> U:
        obj = cls.__new__(cls)
        obj.data = data
        obj.__frozen__ = True
        return obj

    def __update__(self, data: T) -> None:
        object.__setattr__(self, "__frozen__", False)
        self.data = data
        self.__frozen__ = True
