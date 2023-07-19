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
import dataclasses
from abc import abstractmethod
from typing import Dict
from typing import List
from typing import Union

from pydantic.dataclasses import dataclass

from kolena.workflow import TestCase
from kolena.workflow import TestSample

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol


@dataclass(frozen=True)
class SimpleTestCase:
    name: str
    id: int
    version: int
    tags: Dict[str, str] = dataclasses.field(default_factory=dict)
    # test_samples: List[Tuple[TestSample, GroundTruth]] = dataclasses.field(default_factory=list)


_TestCase = Union[TestCase, SimpleTestCase]


class TestRunnable(Protocol):
    @abstractmethod
    def get_test_cases(self) -> List[_TestCase]:
        raise NotImplementedError

    @abstractmethod
    def load_test_samples_by_test_case(self) -> Dict[int, List[TestSample]]:
        raise NotImplementedError
