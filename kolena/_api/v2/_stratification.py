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
from typing import Dict
from typing import List
from typing import Union

from pydantic import field_validator
from pydantic import model_validator
from pydantic.dataclasses import dataclass

from kolena._api.v2._api import GeneralFieldFilter
from kolena._api.v2._testing import SimpleRange
from kolena._api.v2._testing import StratifyFieldSpec


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

    @field_validator("test_cases")
    @classmethod
    def test_case_name_unique(cls, test_cases: List[TestCase]) -> List[TestCase]:
        if len(test_cases) > len({test_case.name for test_case in test_cases}):
            raise ValueError("Test case name must be unique.")
        return test_cases

    @model_validator(mode="after")
    def validate_stratify_field_or_filter(self) -> "Stratification":
        if not self.stratify_fields and not self.filters:
            raise ValueError("Must provide one of 'stratify_fields' or 'filters'")
        return self
