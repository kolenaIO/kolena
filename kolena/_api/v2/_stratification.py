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
from typing import Dict
from typing import List
from typing import Union

from kolena._api.v2._api import GeneralFieldFilter
from kolena._api.v2._testing import SimpleRange
from kolena._api.v2._testing import StratifyFieldSpec
from kolena._utils.pydantic_v1.dataclasses import dataclass


@dataclass(frozen=True)
class CategoricalValue:
    value: Union[str, bool, int, float, None]


@dataclass(frozen=True)
class RangeValue:
    value: SimpleRange


@dataclass(frozen=True)
class QuantileNumericalValue:
    index: int


@dataclass(frozen=True)
class TestCase:
    name: str
    stratification: List[Union[QuantileNumericalValue, CategoricalValue, RangeValue]]
    # QuantileNumericalValue must come before CategoricalValue in Union;
    #   otherwise, from_dict will cast dicts like {value:[undefined], index:[some_value]}
    #   as CategoricalValue since [undefined] is treated as None


@dataclass(frozen=True)
class Stratification:
    name: str
    stratify_fields: List[StratifyFieldSpec]
    test_cases: List[TestCase]
    filters: Union[Dict[str, GeneralFieldFilter], None] = None

    def __post_init__(self) -> None:
        self._check_test_case_name_unique()
        self._validate_stratify_field_or_filter()

    def _check_test_case_name_unique(self) -> None:
        if len(self.test_cases) > len({test_case.name for test_case in self.test_cases}):
            raise ValueError("Test case name must be unique.")

    def _validate_stratify_field_or_filter(self) -> None:
        if not self.stratify_fields and not self.filters:
            raise ValueError("Must provide one of 'stratify_fields' or 'filters'")
