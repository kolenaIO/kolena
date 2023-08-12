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
from abc import abstractmethod
from typing import List
from typing import Tuple
from typing import Union

from kolena._api.v1.core import Dataset
from kolena.workflow import TestCase
from kolena.workflow import TestSample

try:
    from typing import Protocol
except ImportError:
    from typing_extensions import Protocol

_TestCase = Union[TestCase, Dataset.TestCaseData]


class TestRunnable(Protocol):
    @abstractmethod
    def get_test_cases(self) -> List[_TestCase]:
        raise NotImplementedError

    @abstractmethod
    def load_test_samples_by_test_case(self) -> List[Tuple[_TestCase, List[TestSample]]]:
        raise NotImplementedError
