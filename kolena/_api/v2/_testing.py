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
from dataclasses import astuple
from dataclasses import field
from enum import Enum
from typing import List
from typing import Literal
from typing import Optional
from typing import Tuple
from typing import Union

from pydantic import conint
from pydantic import conlist
from pydantic import constr
from pydantic import StrictInt
from pydantic import StrictStr
from pydantic.dataclasses import dataclass

from kolena._api.v2._filter import Filters
from kolena._api.v2._scope import Scopes
from kolena._api.v2._stats import SingleStatsResponse
from kolena.errors import IncorrectUsageError

MAX_BIN_COUNT = 200
MAX_STRATIFICATION_BUCKET_COUNT = 100
RESERVED_FILTER_FORMAT = "[]"


class Path(str, Enum):
    TESTING = "testing"


@dataclass(frozen=True)
class SimpleRange:
    min: Optional[float] = None
    max: Optional[float] = None

    def __post_init__(self) -> None:
        if self.min is not None and self.max is not None:
            if self.min > self.max:
                raise IncorrectUsageError("min value should be less than or equal to max")
        elif self.min is None and self.max is None:
            return

        if self.min is not None:
            if math.isnan(self.min):
                raise IncorrectUsageError("nan is not supported")
            if math.isinf(self.min):
                object.__setattr__(self, "min", None)

        if self.max is not None:
            if math.isnan(self.max):
                raise IncorrectUsageError("nan is not supported")
            if math.isinf(self.max):
                object.__setattr__(self, "max", None)

        if self.min is None and self.max is None:
            raise IncorrectUsageError(f"invalid values {self.min}, {self.max}")


Source = Literal["datapoint", "result"]


@dataclass(frozen=True)
class DatasetField:
    source: Source
    field: constr(min_length=1)

    def is_datapoint(self) -> bool:
        return self.source == "datapoint"

    def is_result(self) -> bool:
        return self.source == "result"


class StratificationType(str, Enum):
    CATEGORICAL = "categorical"
    EQUAL_HEIGHT = "equal-height"
    EQUAL_WIDTH = "equal-width"
    NUMERIC_INTERVAL = "numeric-interval"


@dataclass(frozen=True)
class StratificationInfo(DatasetField):
    value: Union[Optional[Union[StrictStr, StrictInt, bool, Tuple[Optional[float], Optional[float]]]], SimpleRange]
    index: int
    type: StratificationType

    def __post_init__(self) -> None:
        _validate_non_internal_list_field(self.field)

        if self.source == "result":
            fields = self.field.split(".", 2)
            if len(fields) >= 2:
                model_id = fields[0].strip('"')
                eval_config_id = fields[1].strip('"')
                object.__setattr__(self, "field", ".".join([model_id, eval_config_id, *fields[2:]]))

        if isinstance(self.value, tuple):
            if len(self.value) != 2:
                raise ValueError("invalid values", self.value)
            value = SimpleRange(min=self.value[0], max=self.value[1])
            object.__setattr__(self, "value", value)


@dataclass(frozen=True)
class TestCaseStats:
    sample_count: int
    metrics: SingleStatsResponse
    stratification: Optional[List[StratificationInfo]] = None


class BucketSplitType(str, Enum):
    EQUAL_WIDTH = "equal-width"
    EQUAL_HEIGHT = "equal-height"


@dataclass(frozen=True)
class TypedBucketSplit:
    count: conint(strict=True, gt=0, le=MAX_STRATIFICATION_BUCKET_COUNT) = 4
    type: BucketSplitType = BucketSplitType.EQUAL_HEIGHT


@dataclass(frozen=True)
class StratifyFieldSpec(DatasetField):
    values: Optional[List[Union[StrictStr, StrictInt, bool, None]]] = None
    buckets: Union[
        conlist(float, min_length=1, max_length=MAX_STRATIFICATION_BUCKET_COUNT),
        TypedBucketSplit,
        None,
    ] = None

    def __post_init__(self) -> None:
        _validate_non_internal_list_field(self.field)
        _validate_buckets(self.buckets)

    def is_categorical(self) -> bool:
        if self.values or self.buckets is None:
            return True

        return False

    def is_numeric(self) -> bool:
        return not self.is_categorical()


@dataclass(frozen=True)
class TestingRequest:
    filters: Filters
    stratify_fields: List[StratifyFieldSpec] = field(default_factory=list)
    scopes: Optional[Scopes] = None

    def __hash__(self) -> int:
        object_tuple = astuple(self, tuple_factory=tuple)
        serialized_str = str(object_tuple)
        return hash(serialized_str)


@dataclass(frozen=True)
class TestingResponse:
    dataset: TestCaseStats
    test_cases: List[TestCaseStats] = field(default_factory=list)


def _validate_non_internal_list_field(field: str) -> None:
    if RESERVED_FILTER_FORMAT in field:
        raise IncorrectUsageError("unsupported field")


def _validate_buckets(buckets: Union[List[float], TypedBucketSplit, None]) -> None:
    if isinstance(buckets, list):
        # allow [x, x] for single-value-single-bucket "stratification"
        if len(buckets) == 2 and math.isclose(buckets[0], buckets[1]):
            return
        if sorted(set(buckets)) != buckets:
            raise IncorrectUsageError("buckets must be strictly monotonically increasing")
