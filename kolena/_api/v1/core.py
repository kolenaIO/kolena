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
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from pydantic import conlist
from pydantic import constr
from pydantic import StrictBool
from pydantic import StrictStr
from pydantic.dataclasses import dataclass
from typing_extensions import Literal  # for python <3.8 compatibility

from kolena._api.v1.batched_load import BatchedLoad


class Model:
    @dataclass(frozen=True)
    class CreateRequest:
        name: str
        metadata: Dict[str, Any]
        workflow: str

    @dataclass(frozen=True)
    class LoadByNameRequest:
        name: str

    @dataclass(frozen=True)
    class EntityData:
        id: int
        name: str
        metadata: Dict[str, Any]
        workflow: str

    @dataclass(frozen=True)
    class DeleteRequest:
        id: int


@dataclass(frozen=True)
class TestCaseInfo:
    id: int
    name: str
    sample_count: int
    # provided by stratification preview to show what bucket this test-case falls into,
    # e.g. {"count": "low", "gender": "female"}. This info is not available from stratification result.
    membership: Optional[Dict[str, str]] = None


class TestCase:
    @dataclass(frozen=True)
    class CreateRequest:
        name: str
        description: str
        workflow: str

    @dataclass(frozen=True)
    class CreateFromExistingRequest(BatchedLoad.WithLoadUUID):
        test_case_name: str
        test_suite_name: str
        workflow: str
        compute_metrics_where_possible: bool = False
        compute_metrics_models: Optional[List[int]] = None

    @dataclass(frozen=True)
    class CreateFromExistingResponse:
        test_case_id: int
        test_suite_id: int  # newly created version ID

    @dataclass(frozen=True)
    class LoadByNameRequest:
        name: str
        version: Optional[int] = None

    @dataclass(frozen=True)
    class EntityData:
        id: int
        name: str
        version: int
        description: str
        workflow: str

    @dataclass(frozen=True)
    class LoadContentsRequest:
        test_case_id: int

    @dataclass(frozen=True)
    class InitLoadContentsRequest(LoadContentsRequest, BatchedLoad.BaseInitDownloadRequest):
        ...

    @dataclass(frozen=True)
    class EditRequest:
        test_case_id: int
        current_version: int
        description: str
        reset: bool = False

    @dataclass(frozen=True)
    class CompleteEditRequest(EditRequest, BatchedLoad.WithLoadUUID):
        ...

    @dataclass(frozen=True)
    class BulkCreateFromExistingRequest:
        task_handle: str
        workflow: str
        test_suite_name: str
        test_suite_description: str
        test_case_names: List[str]
        uuids: List[str]
        compute_metrics_where_possible: bool = False
        compute_metrics_models: Optional[List[int]] = None

    @dataclass(frozen=True)
    class BulkCreateFromExistingResult:
        test_suite_id: int  # ID of specific version
        test_suite_name: str
        test_suite_description: Optional[str]
        test_cases: List[TestCaseInfo]


class TestSuite:
    @dataclass(frozen=True)
    class CreateRequest:
        name: str
        description: str
        workflow: str
        tags: Optional[List[str]] = None

    @dataclass(frozen=True)
    class LoadByNameRequest:
        name: str
        version: Optional[int] = None

    @dataclass(frozen=True)
    class LoadAllRequest:
        workflow: str
        tags: Optional[List[str]] = None

    @dataclass(frozen=True)
    class EntityData:
        id: int
        name: str
        version: int
        description: str
        test_cases: List[TestCase.EntityData]
        workflow: str
        tags: List[str]

    @dataclass(frozen=True)
    class LoadAllResponse:
        test_suites: List["TestSuite.EntityData"]

    @dataclass(frozen=True)
    class EditRequest:
        test_suite_id: int  # ID of version being edited
        current_version: int
        name: str
        description: str
        test_case_ids: List[int]
        tags: Optional[List[str]] = None  # unique set -- list used to preserve ordering

    @dataclass(frozen=True)
    class DeleteRequest:
        test_suite_id: int


TestSuite.LoadAllResponse.__pydantic_model__.update_forward_refs()


class TestRun:
    @dataclass(frozen=True)
    class MarkCrashedRequest:
        test_run_id: int


CATEGORICAL = "categorical"
NUMERIC = "numeric"


@dataclass(frozen=True)
class Dimension:
    column: Literal["test_sample", "test_sample_metadata", "ground_truth"]
    field: str
    datatype: Literal["categorical", "numeric"]

    def is_categorical(self) -> bool:
        return self.datatype == CATEGORICAL

    def is_numeric(self) -> bool:
        return self.datatype == NUMERIC


@dataclass(frozen=True)
class CategoricalBucket:
    values: Union[List[StrictStr], List[StrictBool]]


class IntervalType(str, Enum):
    RIGHT_OPEN = "right-open"
    CLOSED = "closed"


@dataclass(frozen=True)
class NumericBucket:
    min: float
    max: float
    interval_type: IntervalType = IntervalType.RIGHT_OPEN


@dataclass(frozen=True)
class DimensionSpec:
    # arbitrarily selected max_length for sanity reason
    name: constr(min_length=1, max_length=100)
    columns: conlist(Dimension, min_items=1, max_items=3)
    buckets: Dict[str, List[Union[CategoricalBucket, NumericBucket]]]

    def __post_init_post_parse__(self) -> None:
        n_cols = len(self.columns)
        # auto-stratification only supported for single categorical field
        if not self.buckets and not self.should_auto_stratify():
            raise ValueError("Empty bucket.")

        check_duplicate_fields(self.columns)

        for bucket_name, filters in self.buckets.items():
            if len(filters) != n_cols:
                raise ValueError(f"Bucket '{bucket_name}' does not match number of columns {n_cols}.")

            for dimension, filter in zip(self.columns, filters):
                if (dimension.is_categorical() and not isinstance(filter, CategoricalBucket)) or (
                    dimension.is_numeric() and not isinstance(filter, NumericBucket)
                ):
                    raise ValueError(f"Column {dimension} has incompatible filter.")

    def should_auto_stratify(self) -> bool:
        return not self.buckets and len(self.columns) == 1 and self.columns[0].is_categorical()


@dataclass(frozen=True)
class BaseStratifyRequest:
    workflow: str
    test_case_id: int
    test_suite_name: constr(strict=True, min_length=1)
    # stratification should be non-empty, and currently restrict to max 3 cross-dimension
    strata: conlist(DimensionSpec, min_items=1, max_items=3)

    def __post_init_post_parse__(self) -> None:
        unique_names = {s.name for s in self.strata}
        if len(unique_names) != len(self.strata):
            raise ValueError("Duplicate stratification name")

        check_duplicate_fields([stratum for s in self.strata for stratum in s.columns])


@dataclass(frozen=True)
class StratifyResponse:
    test_suite_id: int  # newly created version ID
    test_suite_name: str
    test_suite_description: str
    base_test_case: TestCaseInfo
    stratified_test_cases: List[TestCaseInfo]


def check_duplicate_fields(strata: List[Dimension]) -> None:
    field_set = set()
    for stratum in strata:
        field = (stratum.column, stratum.field)
        # prevent self-cross stratification, i.e. one field can only be used once
        if field in field_set:
            raise ValueError(f"{field} is already used for stratification.")
        field_set.add(field)
