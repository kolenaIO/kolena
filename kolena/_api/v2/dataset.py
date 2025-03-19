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
from dataclasses import field
from enum import Enum
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

from typing_extensions import Literal

from kolena._api.v1.batched_load import BatchedLoad
from kolena._utils.pydantic_v1 import conint
from kolena._utils.pydantic_v1 import StrictBool
from kolena._utils.pydantic_v1 import StrictStr
from kolena._utils.pydantic_v1.dataclasses import dataclass


class Path(str, Enum):
    REGISTER = "/dataset/register"
    LOAD_DATAPOINTS = "/dataset/load-datapoints"
    LOAD_DATASET = "/dataset/load-by-name"
    LIST_COMMITS = "/dataset/list-commits"
    LIST_DATASETS = "/dataset/list-datasets"


@dataclass(frozen=True)
class RegisterRequest:
    name: str
    id_fields: List[str]
    uuid: str
    sources: Optional[List[Dict[str, str]]]
    append_only: bool = False
    tags: Optional[List[str]] = None
    dataset_tags: Optional[List[str]] = None
    description: Optional[str] = None


@dataclass(frozen=True)
class GeneralFieldFilter:
    """
    Generic representation of a filter on Kolena.
    """

    value_in: Optional[List[Union[StrictStr, StrictBool]]] = None
    """A list of desired categorical values."""
    null_value: Optional[Literal[True]] = None
    """Whether to filter for cases where the field has null value or the field name does not exist."""


@dataclass(frozen=True)
class Filters:
    """
    Filters to be applied on the dataset during the operation. Currently only used as an optional argument
     in [`download_dataset`][kolena.dataset.download_dataset].
    """

    datapoint: Dict[str, GeneralFieldFilter] = field(default_factory=dict)
    """
    Dictionary of a field name of the datapoint to the [`GeneralFieldFilter`][kolena.dataset.GeneralFieldFilter] to be
    applied on the field. In case of nested objects, use `.` as the delimiter to separate the keys. For example, if you
    have a `ground_truth` column of [`Label`][kolena.annotation.Label] type, you can use `ground_truth.label` as the key
    to query for the class label.
    """


@dataclass(frozen=True)
class LoadDatapointsRequest(BatchedLoad.BaseInitDownloadRequest):
    name: str
    commit: Optional[str] = None
    include_extracted_properties: bool = False
    filters: Optional[Filters] = None


@dataclass(frozen=True)
class LoadDatasetByNameRequest:
    name: str
    raise_error_if_not_found: bool = True


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
class ListDatasetsResponse:
    datasets: List[str]


@dataclass(frozen=True)
class ListCommitHistoryResponse:
    records: List[CommitData]
    total_count: int
    descending: bool
    offset: int
    limit: int
