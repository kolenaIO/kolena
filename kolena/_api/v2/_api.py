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
import math
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import root_validator
from pydantic import StrictFloat
from pydantic import StrictInt
from pydantic.dataclasses import dataclass
from pydantic.types import StrictBool
from pydantic.types import StrictStr
from typing_extensions import Literal


@dataclass(frozen=True)
class Range:
    min: float = -math.inf
    max: float = math.inf
    modulo: Optional[int] = None

    @classmethod
    @root_validator(skip_on_failure=True)
    def validate_min_max(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        if values["min"] > values["max"]:
            raise ValueError("Invalid min/max range.")

        return values


@dataclass
class FieldDescriptor:
    value_in: Optional[List[Union[StrictStr, StrictBool]]] = None
    number_range: Optional[Range] = None
    null_value: bool = False
    array_contains: Optional[List[Union[StrictStr, StrictBool, StrictInt, StrictFloat]]] = None
    value_in_distinct_count: Optional[int] = None

    def merge(self, other: "FieldDescriptor") -> "FieldDescriptor":
        return FieldDescriptor(
            value_in=self.value_in or other.value_in,
            number_range=self.number_range or other.number_range,
            null_value=self.null_value or other.null_value,
            array_contains=self.array_contains or other.array_contains,
            value_in_distinct_count=self.value_in_distinct_count or other.value_in_distinct_count,
        )

    def __post_init__(self) -> None:
        if self.value_in is not None and self.value_in_distinct_count is None:
            self.value_in_distinct_count = len(self.value_in)


@dataclass(frozen=True)
class FieldKey:
    key: str
    data: FieldDescriptor


@dataclass(frozen=True)
class GeneralFieldFilterOptions:
    regex_match: bool = False
    regex_flags: str = ""
    inverse: bool = False


@dataclass(frozen=True)
class GeneralFieldFilter:
    value_in: Optional[List[Union[StrictStr, StrictBool]]] = None
    number_range: Optional[Range] = None
    contains: Optional[str] = None
    null_value: Optional[Literal[True]] = None
    array_contains: Optional[List[Union[StrictStr, StrictBool, StrictInt, StrictFloat]]] = None
    options: GeneralFieldFilterOptions = GeneralFieldFilterOptions()

    @classmethod
    @root_validator(skip_on_failure=True)
    def validate_single_operation(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        # check exactly one filter operation is specified
        if sum(1 for value in values.values() if value is not None) != 1:
            raise ValueError("Must provide exactly one operation.")

        return values

    def __hash__(self) -> int:
        return hash((tuple(self.value_in or []), self.number_range, self.contains, self.null_value))
