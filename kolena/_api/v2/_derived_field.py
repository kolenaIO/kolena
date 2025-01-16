# Copyright 2021-2025 Kolena Inc.
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
import re
from typing import Literal
from typing import Optional

from kolena._utils.pydantic_v1 import constr
from kolena._utils.pydantic_v1.dataclasses import dataclass

Source = Literal["datapoint", "result"]

VALID_NAME_PATTERN = r"^[A-Za-z0-9_]+$"


@dataclass(frozen=True)
class CreateRequest:
    dataset_id: int
    name: constr(min_length=1, strip_whitespace=True)
    formula: constr(min_length=1, strip_whitespace=True)
    # This field is intended for internal use and should be set by the system, not included in the request body
    source: Source = "datapoint"

    def __post_init__(self) -> None:
        name = self.name
        if not re.fullmatch(VALID_NAME_PATTERN, name):
            raise ValueError(f"Invalid name: {name}")


@dataclass(frozen=True)
class UpdateRequest:
    name: Optional[constr(min_length=1, strip_whitespace=True)] = None
    formula: Optional[constr(min_length=1, strip_whitespace=True)] = None
    # This field is intended for internal use and should be set by the system, not included in the request body
    source: Optional[Source] = None

    def __post_init__(self) -> None:
        self._check_name()
        self._check_name_formula()

    def _check_name(self) -> None:
        name = self.name
        if name and not re.fullmatch(VALID_NAME_PATTERN, name):
            raise ValueError(f"Invalid name: {name}")

    def _check_name_formula(self) -> None:
        if self.name is None and self.formula is None:
            raise ValueError("Invalid request")


@dataclass(frozen=True)
class ListRequest:
    dataset_id: int


@dataclass(frozen=True)
class DerivedField:
    id: int
    dataset_id: int
    source: Source
    name: str
    formula: str

    def is_datapoint(self) -> bool:
        return self.source == "datapoint"

    def is_result(self) -> bool:
        return self.source == "result"
