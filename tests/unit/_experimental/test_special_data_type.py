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
from typing import Any
from typing import Dict
from typing import Optional

import pytest

from kolena._experimental.special_data_type import _SpecialDataType
from kolena._experimental.special_data_type import Timestamp
from kolena._utils.datatypes import DATA_TYPE_FIELD


@pytest.mark.parametrize(
    "object, json_data",
    [
        (
            Timestamp(epoch_time=1700000000),
            {
                "epoch_time": 1700000000,
                "value": None,
                "format": None,
            },
        ),
        (
            Timestamp(value="12/31/2024, 00:00:00", format="%m/%d/%Y, %H:%M:%S"),
            {
                "epoch_time": 1735603200,
                "value": "12/31/2024, 00:00:00",
                "format": "%m/%d/%Y, %H:%M:%S",
            },
        ),
    ],
)
def test__serde__timestamp(object: Timestamp, json_data: Dict[str, Any]) -> None:
    object_dict = object._to_dict()
    assert object_dict == {
        **json_data,
        DATA_TYPE_FIELD: f"{_SpecialDataType._data_category().value}/{_SpecialDataType.TIMESTAMP.value}",
    }
    assert Timestamp._from_dict(object_dict) == object


@pytest.mark.parametrize(
    "value, format, epoch_time",
    [
        ("12/31/2024, 00:00:00", "%m/%d/%Y, %H:%M:%S", 1735603200),
        ("25/05/99 02:35:5.523", "%d/%m/%y %H:%M:%S.%f", 927599705.523),
        ("2021/05/25", "%Y/%m/%d", 1621900800),
        ("2021-05-25 02:35:15", "%Y-%m-%d %H:%M:%S", 1621910115),
        ("Tuesday, December 31, 2024 5:00:00 AM", "%A, %B %d, %Y %H:%M:%S %p", 1735621200),
        ("Tuesday, December 31, 2024 00:00:00 AM GMT-05:00", "%A, %B %d, %Y %H:%M:%S %p %Z%z", 1735621200),
        ("Tuesday, December 31, 2024 00:00:00 AM UTC-05:00", "%A, %B %d, %Y %H:%M:%S %p %Z%z", 1735621200),
    ],
)
def test__timestamp_epoch_conversion_with_format(value: str, format: str, epoch_time: float) -> None:
    timestamp_object = Timestamp(value=value, format=format)
    assert epoch_time == timestamp_object.epoch_time


@pytest.mark.parametrize(
    "value, epoch_time",
    [
        ("2024-12-31", 1735603200),
        ("2024-12-31 00:00:00", 1735603200),
        ("2024-12-31 12:00:00+00:00", 1735646400),
        ("2024-12-31 12:00:00-00:00", 1735646400),
        ("2024-12-31 12:00:00+05:00", 1735628400),
        ("2024-12-31 12:00:00-05:00", 1735664400),
    ],
)
def test__timestamp_epoch_conversion_iso(value: str, epoch_time: float) -> None:
    timestamp_object = Timestamp(value=value)
    assert epoch_time == timestamp_object.epoch_time


@pytest.mark.parametrize(
    "value, format",
    [
        # value without format and not following ISO 8601 format
        ("12/31/2024, 00:00:00", None),
        # format inconsistent with value
        ("12/31/2024, 00:00:00", "%m/%d/%Y, %s"),
    ],
)
def test__timestamp_validation(value: str, format: Optional[str]) -> None:
    with pytest.raises(ValueError):
        Timestamp(value=value, format=format)
