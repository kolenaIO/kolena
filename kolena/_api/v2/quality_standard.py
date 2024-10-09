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
from datetime import datetime
from enum import Enum
from typing import List

from pydantic.v1 import conint

from kolena._api.v2._metric import MetricGroup
from kolena._api.v2._stratification import Stratification
from kolena._utils.pydantic_v1.dataclasses import dataclass


class Path(str, Enum):
    QUALITY_STANDARD = "quality-standard"
    RESULT = "quality-standard/result"
    COPY_FROM_DATASET = "quality-standard/copy-from-dataset"


@dataclass(frozen=True)
class CopyQualityStandardRequest:
    dataset_id: int
    source_dataset_id: int
    include_metric_groups: bool = True
    include_test_cases: bool = True


@dataclass(frozen=True)
class QualityStandard:
    name: str
    stratifications: List[Stratification]
    metric_groups: List[MetricGroup]

    @classmethod
    def metric_group_name_unique(cls, metric_groups: List[MetricGroup]) -> List[MetricGroup]:
        if len(metric_groups) > len({metric_group.name for metric_group in metric_groups}):
            raise ValueError("Metric group names must be unique.")
        return metric_groups


@dataclass(frozen=True)
class QualityStandardResponse:
    quality_standard_id: conint(gt=0)
    quality_standard: QualityStandard
    updated: datetime
    updated_by: str
