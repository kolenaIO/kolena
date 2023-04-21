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
from typing import List
from typing import Optional

from kolena._api.v1.core import TestCase as API
from kolena._api.v1.workflow import WorkflowType
from kolena.classification import TestCase
from kolena.detection._internal import BaseTestSuite


class TestSuite(BaseTestSuite):
    """
    A test suite is a grouping of :class:`kolena.classification.TestCase` tests.

    Testing on test suites is performed via :mod:`kolena.classification.test`. Metrics are computed across all samples
    in a test suite and also for each individual test case within the suite.

    :param name: the name of the test suite. If a test suite by this name already exists, that test suite is loaded
    :param version: optionally specify the version of the test suite to load. Ignored when the a suite by the
        provided name does not already exist
    :param description: optionally specify a description for the new test suite. Ignored when a test suite with the
        provided name already exists
    :param test_cases: optionally provide a list of :class:`kolena.classification.TestCase` tests used to seed a new
        test suite. Ignored when a test suite with the provided name already exists
    """

    #: Unique name of the test suite.
    name: str

    #: The version of the test suite. Version is automatically incremented whenever the test suite is modified via
    #: :meth:`TestSuite.edit`.
    version: int

    #: Free-form description of this test suite. May be edited at any time via :meth:`TestSuite.edit`.
    description: str

    #: The :class:`kolena.classification.TestCase` tests in this test suite. May be edited at any time via
    #: :meth:`TestSuite.edit`.
    test_cases: List[TestCase]

    _id: int
    _workflow: WorkflowType = WorkflowType.CLASSIFICATION

    def __init__(
        self,
        name: str,
        version: Optional[int] = None,
        description: Optional[str] = None,
        test_cases: Optional[List[TestCase]] = None,
        reset: bool = False,
    ):
        super().__init__(name, WorkflowType.CLASSIFICATION, version, description, test_cases, reset)

    @classmethod
    def _test_case_from(cls, response: API.EntityData) -> TestCase:
        return TestCase._from(response)
