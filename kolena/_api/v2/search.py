# Copyright 2021-2025 Kolena Inc.
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
from typing import List

from kolena._api.v1.batched_load import BatchedLoad
from kolena._utils.pydantic_v1.dataclasses import dataclass


class Path(str, Enum):
    EMBEDDINGS = "/search/embeddings"
    GET_EMBEDDING_KEYS = "/search/get-embedding-model-keys"
    LOAD_EMBEDDINGS = "/search/load-embeddings"


@dataclass(frozen=True)
class UploadDatasetEmbeddingsRequest(BatchedLoad.WithLoadUUID):
    name: str
    run_embedding_reduction: bool


@dataclass(frozen=True)
class UploadDatasetEmbeddingsResponse:
    n_datapoints: int


@dataclass(frozen=True)
class DownloadDatasetEmbeddingsRequest(BatchedLoad.BaseInitDownloadRequest):
    dataset: str
    model_key: str


@dataclass(frozen=True)
class GetEmbeddingKeysRequest:
    dataset_identifier: str


@dataclass(frozen=True)
class GetEmbeddingKeysResponse:
    model_keys: List[str]
