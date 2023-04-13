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

from pydantic.dataclasses import dataclass

from kolena._api.v1.batched_load import BatchedLoad
from kolena._api.v1.workflow import WorkflowType


class TestImage:
    class Path(str, Enum):
        INIT_LOAD_IMAGES = "/detection/test-sample/load-images/init"

    @dataclass(frozen=True)
    class LoadImagesRequest:
        dataset: Optional[str] = None

    @dataclass(frozen=True)
    class InitLoadImagesRequest(LoadImagesRequest, BatchedLoad.BaseInitDownloadRequest):
        ...


class Model:
    class Path(str, Enum):
        CREATE = "/detection/model/create"
        LOAD_BY_NAME = "/detection/model/load-by-name"
        INIT_LOAD_INFERENCES = "/detection/model/load-inferences/init"
        INIT_LOAD_INFERENCES_BY_TEST_CASE = "/detection/model/load-test-suite-test-cases-inferences/init"
        DELETE = "/detection/model/delete"

    @dataclass(frozen=True)
    class LoadInferencesRequest:
        model_id: int
        batch_size: int
        test_case_id: Optional[int] = None
        test_suite_id: Optional[int] = None

        def __post_init__(self) -> None:
            if (self.test_case_id is None) == (self.test_suite_id is None):
                raise ValueError("must specify exactly one test case or test suite ID")

    @dataclass(frozen=True)
    class InitLoadInferencesRequest(LoadInferencesRequest, BatchedLoad.BaseInitDownloadRequest):
        ...

    @dataclass(frozen=True)
    class LoadInferencesByTestCaseRequest:
        model_id: int
        batch_size: int
        test_suite_id: int

    @dataclass(frozen=True)
    class InitLoadInferencesByTestCaseRequest(
        LoadInferencesByTestCaseRequest,
        BatchedLoad.BaseInitDownloadRequest,
    ):
        ...


class TestCase:
    class Path(str, Enum):
        CREATE = "/detection/test-case/create"
        LOAD_BY_NAME = "/detection/test-case/load-by-name"
        INIT_LOAD_IMAGES = "/detection/test-case/load-images/init"
        COMPLETE_EDIT = "/detection/test-case/edit/complete"


class TestSuite:
    class Path(str, Enum):
        CREATE = "/detection/test-suite/create"
        LOAD_BY_NAME = "/detection/test-suite/load-by-name"
        EDIT = "/detection/test-suite/edit"
        DELETE = "/detection/test-suite/delete"


CustomMetricValue = Union[float, int, None]
CustomMetrics = Dict[str, CustomMetricValue]


class TestRun:
    class Path(str, Enum):
        CREATE_OR_RETRIEVE = "/detection/test-run/create-or-retrieve"
        MARK_CRASHED = "/detection/test-run/mark-crashed"
        INIT_LOAD_REMAINING_IMAGES = "/detection/test-run/load-remaining-images/init"
        UPLOAD_IMAGE_RESULTS = "/detection/test-run/upload-inferences/complete"
        UPLOAD_CUSTOM_METRICS = "/detection/test-run/custom-metrics"

    @dataclass(frozen=True)
    class CreateOrRetrieveRequest:
        model_id: int
        test_suite_ids: List[int]
        config: Optional["Metrics.RunConfig"] = None

    @dataclass(frozen=True)
    class CreateOrRetrieveResponse:
        test_run_id: int

    @dataclass(frozen=True)
    class LoadRemainingImagesRequest:
        test_run_id: int
        batch_size: int

    @dataclass(frozen=True)
    class InitLoadRemainingImagesRequest(LoadRemainingImagesRequest, BatchedLoad.BaseInitDownloadRequest):
        load_all: bool = False

    @dataclass(frozen=True)
    class UploadImageResultsRequest(BatchedLoad.WithLoadUUID):
        test_run_id: int
        reset: bool = False

    @dataclass(frozen=True)
    class UploadImageResultsResponse:
        n_uploaded: int

    @dataclass(frozen=True)
    class UpdateCustomMetricsRequest:
        model_id: int
        metrics: Dict[int, Dict[int, CustomMetrics]]  # testsuite_id -> testcase_id -> CustomMetrics


# work around pydantic nastiness in some environments by declaring this base API object
class _Metrics:
    class RunStrategy(str, Enum):
        F1_OPTIMAL = "F1_OPTIMAL"
        ACCURACY_OPTIMAL = "ACCURACY_OPTIMAL"
        FIXED_GLOBAL_THRESHOLD = "FIXED_GLOBAL_THRESHOLD"


class Metrics(_Metrics):
    @dataclass(frozen=True)
    class RunConfig:
        strategy: _Metrics.RunStrategy
        iou_threshold: float
        params: Optional[Dict[str, Any]] = None

    @dataclass(frozen=True)
    class ComputeRequest:
        test_run_id: int
        workflow_type: WorkflowType
        config: Optional[List["Metrics.RunConfig"]]


# allow referencing e.g. Metrics.RunConfig as type from Metrics.ComputeRequest
Metrics.ComputeRequest.__pydantic_model__.update_forward_refs()  # type: ignore
TestRun.CreateOrRetrieveRequest.__pydantic_model__.update_forward_refs()  # type: ignore
