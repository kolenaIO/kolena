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
