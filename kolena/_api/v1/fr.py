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

from pydantic.dataclasses import dataclass

from kolena._api.v1.batched_load import BatchedLoad


class TestImages:
    class Path(str, Enum):
        INIT_LOAD_REQUEST = "/fr/test-images/load/init"
        COMPLETE_REGISTER = "/fr/test-images/register/complete"

    @dataclass(frozen=True)
    class LoadRequest:
        include_augmented: bool
        data_source: Optional[str] = None
        test_case_id: Optional[int] = None
        test_suite_id: Optional[int] = None

        def __post_init__(self) -> None:
            if sum(1 for value in [self.data_source, self.test_case_id, self.test_suite_id] if value is not None) > 1:
                raise ValueError("must populate at most one of: data_source, test_case_id, test_suite_id")

    @dataclass(frozen=True)
    class InitLoadRequest(LoadRequest, BatchedLoad.BaseInitDownloadRequest):
        ...

    @dataclass(frozen=True)
    class RegisterResponse:
        n_images_inserted: int
        n_images_updated: int
        n_tags_inserted: int
        n_tags_updated: int


class TestCase:
    class Path(str, Enum):
        CREATE = "/fr/test-case/create"
        LOAD_BY_NAME = "/fr/test-case/load-by-name"
        COMPLETE_EDIT = "/fr/test-case/edit/complete"
        INIT_LOAD_DATA = "/fr/test-case/load-data/init"

    @dataclass(frozen=True)
    class CreateRequest:
        name: str
        description: str

    @dataclass(frozen=True)
    class LoadByNameRequest:
        name: str
        version: Optional[int] = None

    @dataclass(frozen=True)
    class LoadDataRequest:
        test_case_id: int

    @dataclass(frozen=True)
    class InitLoadDataRequest(LoadDataRequest, BatchedLoad.BaseInitDownloadRequest):
        ...

    @dataclass(frozen=True)
    class EditRequest:
        test_case_id: int
        current_version: int
        name: str  # TODO: deprecate and remove (unused)
        description: str

    @dataclass(frozen=True)
    class CompleteEditRequest(EditRequest, BatchedLoad.WithLoadUUID):
        ...

    @dataclass(frozen=True)
    class EntityData:
        id: int
        name: str
        version: int
        description: str
        image_count: int
        pair_count_genuine: int
        pair_count_imposter: int


class TestSuite:
    class Path(str, Enum):
        CREATE = "/fr/test-suite/create"
        LOAD_BY_NAME = "/fr/test-suite/load-by-name"
        EDIT = "/fr/test-suite/edit"
        DELETE = "/fr/test-suite/delete"

    @dataclass(frozen=True)
    class CreateRequest:
        name: str
        description: str

    @dataclass(frozen=True)
    class LoadByNameRequest:
        name: str
        version: Optional[int] = None

    @dataclass(frozen=True)
    class EditRequest:
        test_suite_id: int
        current_version: int
        name: str
        description: str
        baseline_test_case_ids: List[int]
        non_baseline_test_case_ids: List[int]

    @dataclass(frozen=True)
    class EntityData:
        id: int
        name: str
        version: int
        description: str
        baseline_test_cases: List[TestCase.EntityData]
        non_baseline_test_cases: List[TestCase.EntityData]
        baseline_image_count: int
        baseline_pair_count_genuine: int
        baseline_pair_count_imposter: int

    @dataclass(frozen=True)
    class DeleteRequest:
        test_suite_id: int


class Model:
    class Path(str, Enum):
        CREATE = "/fr/model/create"
        LOAD_BY_NAME = "/fr/model/load-by-name"
        INIT_LOAD_PAIR_RESULTS = "/fr/model/load-pair-results/init"
        DELETE = "/fr/model/delete"

    @dataclass(frozen=True)
    class LoadByNameRequest:
        name: str

    @dataclass(frozen=True)
    class CreateRequest:
        name: str
        metadata: Dict[str, Any]

    @dataclass(frozen=True)
    class LoadPairResultsRequest:
        model_id: int
        # exactly one of the two must be present
        test_suite_id: Optional[int] = None
        test_case_id: Optional[int] = None

        def __post_init_post_parse__(self) -> None:
            if self.test_suite_id is None and self.test_case_id is None:
                raise ValueError("either test_case_id or test_suite_id must be present")
            if self.test_suite_id is not None and self.test_case_id is not None:
                raise ValueError("only one of test_case_id or test_suite_id may be present")

    @dataclass(frozen=True)
    class InitLoadPairResultsRequest(LoadPairResultsRequest, BatchedLoad.BaseInitDownloadRequest):
        ...

    @dataclass(frozen=True)
    class DeleteRequest:
        model_id: int


class TestRun:
    class Path(str, Enum):
        CREATE_OR_RETRIEVE = "/fr/test-run/create-or-retrieve"
        INIT_LOAD_REMAINING_IMAGES = "/fr/test-run/load-remaining-images/init"
        COMPLETE_UPLOAD_IMAGE_RESULTS = "/fr/test-run/upload-image-results/complete"
        INIT_LOAD_REMAINING_PAIRS = "/fr/test-run/load-remaining-pairs/init"
        COMPLETE_UPLOAD_PAIR_RESULTS = "/fr/test-run/upload-pair-results/complete"
        MARK_CRASHED = "/fr/test-run/mark-crashed"

    @dataclass(frozen=True)
    class CreateOrRetrieveRequest:
        model_id: int
        test_suite_ids: List[int]
        reset: bool = False

    @dataclass(frozen=True)
    class MarkCrashedRequest:
        test_run_id: int

    @dataclass(frozen=True)
    class LoadRemainingImagesRequest:
        test_run_id: int
        batch_size: int
        load_all: bool = False

    @dataclass(frozen=True)
    class InitLoadRemainingImagesRequest(LoadRemainingImagesRequest, BatchedLoad.BaseInitDownloadRequest):
        ...

    @dataclass(frozen=True)
    class UploadImageResultsRequest(BatchedLoad.WithLoadUUID):
        test_run_id: int
        reset: bool = False

    @dataclass(frozen=True)
    class UploadImageResultsResponse:
        n_uploaded: int

    @dataclass(frozen=True)
    class LoadRemainingPairsRequest:
        test_run_id: int
        batch_size: int
        load_all: bool = False

    @dataclass(frozen=True)
    class InitLoadRemainingPairsRequest(LoadRemainingPairsRequest, BatchedLoad.BaseInitDownloadRequest):
        ...

    @dataclass(frozen=True)
    class InitLoadRemainingPairsPartialResponse:
        pairs: BatchedLoad.InitDownloadPartialResponse
        embeddings: BatchedLoad.InitDownloadPartialResponse

    @dataclass(frozen=True)
    class UploadPairResultsRequest(BatchedLoad.WithLoadUUID):
        test_run_id: int
        reset: bool = False

    @dataclass(frozen=True)
    class UploadPairResultsResponse:
        n_uploaded: int


class Asset:
    class Path(str, Enum):
        CONFIG = "/fr/asset/config"
        BULK_UPLOAD = "/fr/asset/upload/bulk"

    @dataclass(frozen=True)
    class Config:
        bucket: str
        prefix: str

    @dataclass(frozen=True)
    class BulkUploadResponse:
        n_uploaded: int
