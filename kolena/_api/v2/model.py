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
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from kolena._api.v1.batched_load import BatchedLoad
from kolena._utils.pydantic_v1.dataclasses import dataclass


class Path(str, Enum):
    UPLOAD_RESULTS = "/model/upload-results"
    LOAD_RESULTS = "/model/load-results"


@dataclass(frozen=True)
class LoadByNameRequest:
    name: str


@dataclass(frozen=True)
class IdRequest:
    id: int


@dataclass(frozen=True)
class LoadResultsRequest(BatchedLoad.BaseInitDownloadRequest):
    model: str
    dataset: str
    commit: Optional[str] = None
    include_extracted_properties: bool = False


@dataclass(frozen=True)
class UploadResultsRequest:
    model: str
    uuid: str
    dataset_id: int
    sources: Optional[List[Dict[str, str]]]


@dataclass(frozen=True)
class UploadResultsResponse:
    n_inserted: int
    n_updated: int
    model_id: int
    eval_config_id: Union[int, List[int]]


@dataclass(frozen=True)
class EntityData:
    id: int
    name: str
