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

from pydantic import conint
from pydantic.dataclasses import dataclass

from kolena._api.v1.batched_load import BatchedLoad


class Path(str, Enum):
    REGISTER = "/dataset/register"
    LOAD_DATAPOINTS = "/dataset/load-datapoints"
    LOAD_DATASET = "/dataset/load-by-name"
    LIST_COMMITS = "/dataset/list-commits"


@dataclass(frozen=True)
class RegisterRequest:
    name: str
    id_fields: List[str]
    uuid: str
    sources: Optional[List[Dict[str, str]]]


@dataclass(frozen=True)
class LoadDatapointsRequest(BatchedLoad.BaseInitDownloadRequest):
    name: str
    commit: Optional[str] = None


@dataclass(frozen=True)
class LoadDatasetByNameRequest:
    name: str


@dataclass(frozen=True)
class EntityData:
    id: int
    name: str
    description: str
    id_fields: List[str]


@dataclass(frozen=True)
class ListCommitHistoryRequest:
    name: str
    descending: bool = False
    offset: conint(strict=True, ge=0) = 0
    limit: conint(strict=True, ge=0, le=100) = 50


@dataclass(frozen=True)
class CommitData:
    commit: str
    timestamp: int
    user: str
    n_removed: int
    n_added: int


@dataclass(frozen=True)
class ListCommitHistoryResponse:
    records: List[CommitData]
    total_count: int
    descending: bool
    offset: int
    limit: int
