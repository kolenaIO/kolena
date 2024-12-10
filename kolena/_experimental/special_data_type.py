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
"""
Special data types supported on the Kolena platform.

"""  # noqa: E501
import os
from abc import ABCMeta
from datetime import datetime
from typing import Optional

from kolena._utils.datatypes import DataCategory
from kolena._utils.datatypes import DataType
from kolena._utils.datatypes import TypedDataObject
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.validators import ValidatorConfig

os.putenv("TZ", "GMT")


class _SpecialDataType(DataType):
    TIMESTAMP = "TIMESTAMP"

    @staticmethod
    def _data_category() -> DataCategory:
        return DataCategory.SPECIAL


@dataclass(frozen=True, config=ValidatorConfig)
class SpecialDataType(TypedDataObject[_SpecialDataType], metaclass=ABCMeta):
    """The base class for all special data types."""


@dataclass(frozen=True, config=ValidatorConfig)
class Timestamp(SpecialDataType):
    """
    !!! note "Experimental"
        This class is considered **experimental**

    Timestamp data type.
    """

    epoch_time: Optional[float] = None
    """The epoch time of the timestamp. If `value` and `format` are specified, the `epoch_time` will be calculated."""

    value: Optional[str] = None
    """
    The timestamp in a string representation. If present, the corresponding `format` must be specified too.
    Note that GMT timezone is assumed unless the offset is specified in the string.
    """

    format: Optional[str] = None
    """
    The format of the `value` string following the
    [python format codes](https://docs.python.org/3/library/datetime.html#strftime-and-strptime-format-codes).
    """

    @staticmethod
    def _data_type() -> _SpecialDataType:
        return _SpecialDataType.TIMESTAMP

    def __post_init__(self) -> None:
        if self.value:
            if not self.format:
                raise ValueError("format needs to be specified for string timestamp")
            object.__setattr__(
                self,
                "epoch_time",
                datetime.strptime(self.value, self.format).timestamp(),
            )
