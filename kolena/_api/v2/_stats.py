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
from dataclasses import field
from typing import Dict
from typing import List
from typing import Literal
from typing import Union

from pydantic import StrictBool
from pydantic import StrictStr
from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CategoricalBucket:
    value: Union[StrictStr, StrictBool, int, None]
    count: int


@dataclass(frozen=True)
class NumericBucket:
    lower: float
    upper: float
    count: int
    interval_type: Literal["closed", "right-open"] = "right-open"


@dataclass(frozen=True)
class CategoricalStats:
    histogram: List[CategoricalBucket]
    data_type: Literal["categorical"] = "categorical"


@dataclass(frozen=True)
class NumericStats:
    count: int
    mean: float
    median: float
    min: float
    max: float
    stddev: float
    sum: float
    histogram: List[NumericBucket]
    data_type: Literal["numeric"] = "numeric"


@dataclass(frozen=True)
class ArraySizeStats:
    mean: float
    median: float
    min: int
    max: int
    stddev: float
    sum: int
    histogram: List[CategoricalBucket]


@dataclass(frozen=True)
class ArrayStats:
    length: ArraySizeStats
    data_type: Literal["array"] = "array"


@dataclass(frozen=True)
class FieldStats:
    key: str
    data: Union[CategoricalStats, NumericStats, ArrayStats]


@dataclass(frozen=True)
class SingleStatsResponse:
    datapoint: List[FieldStats] = field(default_factory=list)
    # key by model id first, then eval_config_id
    result: Dict[str, Dict[str, List[FieldStats]]] = field(default_factory=dict)
    llm: List[FieldStats] = field(default_factory=list)


@dataclass(frozen=True)
class FieldStatsResponse:
    stats: List[SingleStatsResponse]
