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

from kolena._api.v1.batched_load import BatchedLoad
from kolena._utils.pydantic_v1.dataclasses import dataclass


class Model:
    class Path(str, Enum):
        CREATE = "/generic/model/create"
        LOAD = "/generic/model/load"
        LOAD_ALL = "/generic/model/load-all"
        DELETE = "/generic/model/delete"
        LOAD_INFERENCES = "/generic/model/load-inferences"

    @dataclass(frozen=True)
    class LoadInferencesRequest(BatchedLoad.BaseInitDownloadRequest):
        model_id: int
        test_case_id: int


class TestCase:
    class Path(str, Enum):
        CREATE = "/generic/test-case/create"
        LOAD = "/generic/test-case/load"
        INIT_LOAD_TEST_SAMPLES = "/generic/test-case/load-test-samples"
        COMPLETE_EDIT = "/generic/test-case/edit"
        BULK_PROCESS = "/generic/test-case/bulk-process"


class TestSuite:
    class Path(str, Enum):
        CREATE = "/generic/test-suite/create"
        LOAD = "/generic/test-suite/load"
        LOAD_ALL = "/generic/test-suite/load-all"
        EDIT = "/generic/test-suite/edit"
        DELETE = "/generic/test-suite/delete"
        INIT_LOAD_TEST_SAMPLES = "/generic/test-suite/load-test-samples"

    @dataclass(frozen=True)
    class LoadTestSamplesRequest(BatchedLoad.BaseInitDownloadRequest):
        test_suite_id: int


class TestRun:
    class Path(str, Enum):
        CREATE_OR_RETRIEVE = "/generic/test-run/create-or-retrieve"
        MARK_CRASHED = "/generic/test-run/mark-crashed"
        EVALUATE = "/generic/test-run/evaluate"
        LOAD_TEST_SAMPLES = "/generic/test-run/load-remaining-test-samples"
        LOAD_INFERENCES = "/generic/test-run/load-test-sample-inferences"
        UPLOAD_INFERENCES = "/generic/test-run/upload-inferences"
        UPLOAD_TEST_SAMPLE_METRICS = "/generic/test-run/upload-test-sample-metrics"
        UPLOAD_TEST_SAMPLE_METRICS_THRESHOLDED = "/generic/test-run/upload-test-sample-thresholded-metrics"
        UPLOAD_TEST_CASE_METRICS = "/generic/test-run/upload-test-case-metrics"
        UPLOAD_TEST_CASE_PLOTS = "/generic/test-run/upload-test-case-plots"
        UPLOAD_TEST_SUITE_METRICS = "/generic/test-run/upload-test-suite-metrics"
        UPDATE_METRICS_STATUS = "/generic/test-run/update-metrics-status"

    @dataclass(frozen=True)
    class EvaluatorConfiguration:
        display_name: str
        configuration: Dict[str, Any]  # TODO: real type is JSON

    @dataclass(frozen=True)
    class CreateOrRetrieveRequest:
        model_id: int
        test_suite_id: int
        evaluator: Optional[str] = None
        configurations: Optional[List["TestRun.EvaluatorConfiguration"]] = None

    @dataclass(frozen=True)
    class CreateOrRetrieveResponse:
        test_run_id: int

    @dataclass(frozen=True)
    class EvaluateRequest:
        test_run_id: int

    EvaluateResponse = CreateOrRetrieveResponse

    @dataclass(frozen=True)
    class LoadRemainingTestSamplesRequest(BatchedLoad.BaseInitDownloadRequest):
        test_run_id: int
        load_all: bool = False

    @dataclass(frozen=True)
    class LoadTestSampleInferencesRequest(BatchedLoad.BaseInitDownloadRequest):
        test_run_id: int

    @dataclass(frozen=True)
    class UploadInferencesRequest(BatchedLoad.WithLoadUUID):
        test_run_id: int
        reset: bool = False

    @dataclass(frozen=True)
    class UploadTestSampleMetricsRequest(BatchedLoad.WithLoadUUID):
        test_run_id: int
        test_case_id: Optional[int]
        configuration: Optional["TestRun.EvaluatorConfiguration"]

    @dataclass(frozen=True)
    class UploadTestSampleThresholdedMetricsRequest(BatchedLoad.WithLoadUUID):
        test_run_id: int
        test_case_id: Optional[int]
        configuration: Optional["TestRun.EvaluatorConfiguration"]
        model_id: int

    @dataclass(frozen=True)
    class UpdateMetricsStatusRequest:
        test_run_id: int
        progress: float
        message: str = ""

    @dataclass(frozen=True)
    class UploadAggregateMetricsRequest(BatchedLoad.WithLoadUUID):
        test_run_id: int
        test_suite_id: int

    @dataclass(frozen=True)
    class BulkUploadResponse:
        n_uploaded: int


class Workflow:
    class Path(str, Enum):
        EVALUATOR = "/generic/workflow/evaluator"

    @dataclass(frozen=True)
    class EvaluatorRoleConfig:
        job_role_name: str
        job_role_arn: str
        external_id: str
        assume_role_arn: str

    @dataclass(frozen=True)
    class RegisterEvaluatorRequest:
        workflow: str
        name: str
        image: str
        secret: Optional[str] = None
        aws_assume_role: Optional[str] = None

    @dataclass(frozen=True)
    class EvaluatorResponse:
        name: str
        image: Optional[str] = None
        created: Optional[str] = None
        secret: Optional[str] = None
        aws_role_config: Optional["Workflow.EvaluatorRoleConfig"] = None

    @dataclass(frozen=True)
    class ListEvaluatorsResponse:
        evaluators: List["Workflow.EvaluatorResponse"]


class Search:
    class Path(str, Enum):
        EMBEDDINGS = "/generic/search/embeddings"

    @dataclass(frozen=True)
    class UploadEmbeddingsRequest(BatchedLoad.WithLoadUUID):
        ...

    @dataclass(frozen=True)
    class UploadEmbeddingsResponse:
        n_samples: int


TestRun.CreateOrRetrieveRequest.__pydantic_model__.update_forward_refs()  # type: ignore
TestRun.UploadTestSampleMetricsRequest.__pydantic_model__.update_forward_refs()  # type: ignore
TestRun.UploadTestSampleThresholdedMetricsRequest.__pydantic_model__.update_forward_refs()  # type: ignore
Workflow.EvaluatorResponse.__pydantic_model__.update_forward_refs()  # type: ignore
Workflow.ListEvaluatorsResponse.__pydantic_model__.update_forward_refs()  # type: ignore
