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
from kolena.detection import TestCase
from kolena.detection._internal import BaseTestSuite


class TestSuite(BaseTestSuite):
    """
    A test suite groups together one or more test cases.

    For additional functionality, see the associated
    [base class documentation][kolena.detection._internal.test_suite.BaseTestSuite].

    :param name: Name of the test suite to create or load.
    :param version: Optionally specify the version of the test suite to load. When absent, the latest version is loaded.
        Ignored when creating new test suites.
    :param description: Optionally specify a description for a newly created test suite. For existing test suites, this
        description can be edited via [`TestSuite.edit`][kolena.detection._internal.test_suite.BaseTestSuite.edit].
    :param test_cases: Optionally specify a list of test cases to populate a new test suite. For existing test suites,
        test cases can be edited via [`TestSuite.edit`][kolena.detection._internal.test_suite.BaseTestSuite.edit].
    """

    name: str
    """Unique name for this test suite."""

    version: int
    """
    The version of the test suite. This version is automatically incremented each time the suite is edited with
    [`TestSuite.edit`][kolena.detection._internal.test_suite.BaseTestSuite.edit].
    """

    description: str
    """
    Free-form, human-readable description of the test suite. Can be edited at any time via
    [`TestSuite.edit`][kolena.detection._internal.test_suite.BaseTestSuite.edit].
    """

    test_cases: List[TestCase]
    """The test cases within this test suite."""

    _id: int
    _workflow: WorkflowType = WorkflowType.DETECTION

    def __init__(
        self,
        name: str,
        version: Optional[int] = None,
        description: Optional[str] = None,
        test_cases: Optional[List[TestCase]] = None,
        reset: bool = False,
    ):
        super().__init__(name, WorkflowType.DETECTION, version, description, test_cases, reset)

    @classmethod
    def _test_case_from(cls, response: API.EntityData) -> TestCase:
        return TestCase._from(response)
