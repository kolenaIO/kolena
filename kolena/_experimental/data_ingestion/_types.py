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
from uuid import uuid4

from pydantic import conlist
from pydantic.dataclasses import dataclass


def _random_name(prefix: str) -> str:
    return f"{prefix}-{uuid4()}"


@dataclass
class DataIngestionConfig:
    data_paths: conlist(str, min_items=1)
    locator_prefix: str = ""
    test_case_name: str = _random_name("test-case")
    test_suite_name: str = _random_name("test-suite")
    model_name: str = _random_name("model")
    reset: bool = False
