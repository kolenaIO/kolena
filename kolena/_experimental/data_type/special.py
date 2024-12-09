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
from abc import ABCMeta
from datetime import datetime
from typing import Optional

from kolena._utils.datatypes import DataCategory
from kolena._utils.datatypes import DataType
from kolena._utils.datatypes import TypedDataObject
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.validators import ValidatorConfig


class _SpecialDataType(DataType):
    TIMESTAMP = "TIMESTAMP"

    @staticmethod
    def _data_category() -> DataCategory:
        return DataCategory.ANNOTATION


@dataclass(frozen=True, config=ValidatorConfig)
class Annotation(TypedDataObject[_SpecialDataType], metaclass=ABCMeta):
    """The base class for all special data types."""


@dataclass(frozen=True, config=ValidatorConfig)
class Timestamp(Annotation):
    """
    !!! note "Experimental"

        Timestamp data type.
    """

    epoch_time: Optional[float] = None
    value: Optional[str] = None
    format: Optional[str] = None

    @staticmethod
    def _data_type() -> _SpecialDataType:
        return _SpecialDataType.TIMESTAMP

    # TODO: unit tests
    def __post_init__(self) -> None:
        if self.value:
            if not self.format:
                raise ValueError("format needs to be specified for string timestamp")
            object.__setattr__(self, "epoch_time", datetime.strptime(self.value, self.format).timestamp())
