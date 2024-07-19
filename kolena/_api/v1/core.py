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
from enum import Enum
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Set

from kolena._api.v1.batched_load import BatchedLoad
from kolena._utils.pydantic_v1.dataclasses import dataclass


class Model:
    @dataclass(frozen=True)
    class CreateRequest:
        name: str
        metadata: Dict[str, Any]
        workflow: str
        tags: Optional[List[str]] = None

    @dataclass(frozen=True)
    class LoadByNameRequest:
        name: str

    @dataclass(frozen=True)
    class LoadAllRequest:
        workflow: str
        tags: Optional[List[str]] = None

    @dataclass(frozen=True)
    class EntityData:
        id: int
        name: str
        metadata: Dict[str, Any]
        tags: Set[str]
        workflow: str

    @dataclass(frozen=True)
    class LoadAllResponse:
        models: List["Model.EntityData"]

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


class BulkProcessStatus(Enum):
    CREATED = "created"
    LOADED = "loaded"
    EDITED = "edited"


class TestCase:
    @dataclass(frozen=True)
    class CreateRequest:
        name: str
        description: str
        workflow: str

    @dataclass(frozen=True)
    class SingleProcessRequest:
        name: str
        reset: bool = False

    @dataclass(frozen=True)
    class SingleProcessResponse:
        data: "TestCase.EntityData"
        status: BulkProcessStatus

    @dataclass(frozen=True)
    class BulkProcessRequest:
        test_cases: List["TestCase.SingleProcessRequest"]
        workflow: str
        uuid: Optional[str] = None

    @dataclass(frozen=True)
    class BulkProcessResponse:
        test_cases: List["TestCase.SingleProcessResponse"]

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


class TestRun:
    @dataclass(frozen=True)
    class MarkCrashedRequest:
        test_run_id: int


Model.LoadAllResponse.__pydantic_model__.update_forward_refs()  # type: ignore
TestCase.SingleProcessResponse.__pydantic_model__.update_forward_refs()  # type: ignore[attr-defined]
TestCase.BulkProcessRequest.__pydantic_model__.update_forward_refs()  # type: ignore[attr-defined]
TestCase.BulkProcessResponse.__pydantic_model__.update_forward_refs()  # type: ignore[attr-defined]
TestSuite.LoadAllResponse.__pydantic_model__.update_forward_refs()  # type: ignore
