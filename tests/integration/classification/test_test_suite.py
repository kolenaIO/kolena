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

import pytest
from integration.helper import fake_random_locator

from kolena._api.v1.workflow import WorkflowType
from kolena.classification import TestCase
from kolena.classification import TestImage
from kolena.classification import TestSuite
from kolena.detection import TestSuite as DetectionTestSuite
from kolena.errors import NotFoundError
from kolena.errors import WorkflowMismatchError
from tests.integration.helper import with_test_prefix


@pytest.fixture(scope="module")
def test_case() -> TestCase:
    name = with_test_prefix(f"{__file__}::test_case fixture test case")
    return TestCase(name, description="test case description")


@pytest.fixture(scope="module")
def test_dataset() -> List[TestImage]:
    name = with_test_prefix(f"{__file__}::test_dataset fixture dataset")
    return [
        TestImage(fake_random_locator(), dataset=name),
        TestImage(fake_random_locator(), dataset=name, labels=["car"]),
    ]


@pytest.fixture(scope="module")
def test_case_versions(test_dataset: List[TestImage]) -> List[TestCase]:
    name = with_test_prefix(f"{__file__}::test_case_versions fixture test case")
    test_case = TestCase(name, description="test case description")
    # load copy such that it is not modified by later edits
    test_case_v0 = TestCase(test_case.name, version=test_case.version)
    with test_case.edit() as editor:
        editor.add(test_dataset[0])
    test_case_v1 = TestCase(test_case.name, version=test_case.version)
    with test_case.edit() as editor:
        editor.add(test_dataset[1])
    test_case_v2 = TestCase(test_case.name, version=test_case.version)
    return [test_case_v0, test_case_v1, test_case_v2]


def test__create() -> None:
    test_suite_name = with_test_prefix(f"{__file__} test__create test suite")
    description = "A\n\tlong\ndescription including special characters! 🎉"
    test_suite = TestSuite.create(test_suite_name, description=description)
    assert test_suite.name == test_suite_name
    assert test_suite.version == 0
    assert test_suite.description == description
    assert test_suite.test_cases == []
    assert test_suite._workflow == WorkflowType.CLASSIFICATION


def test__create__with_test_cases(test_case: TestCase, test_case_versions: List[TestCase]) -> None:
    test_suite_name = with_test_prefix(f"{__file__} test__create__with_test_cases test suite")
    description = "A\n\tlong\ndescription including special characters! 🎉"
    test_cases = [test_case, test_case_versions[0]]
    test_suite = TestSuite.create(test_suite_name, description=description, test_cases=test_cases)
    assert test_suite.name == test_suite_name
    assert test_suite.version == 1
    assert test_suite.description == description
    assert test_suite.test_cases == test_cases
    assert test_suite._workflow == WorkflowType.CLASSIFICATION


def test__load() -> None:
    test_suite_name = with_test_prefix(f"{__file__} test__load test suite")
    test_suite = TestSuite(test_suite_name)
    loaded_test_suite = TestSuite.load(test_suite_name)
    for key in ["name", "version", "description", "test_cases", "_id", "_workflow"]:
        assert getattr(test_suite, key) == getattr(loaded_test_suite, key)


def test__load__with_version(test_case_versions: List[TestCase]) -> None:
    test_suite_name = with_test_prefix(f"{__file__} test__load__version test suite")
    test_suite = TestSuite(test_suite_name)
    new_description = f"{__file__} test__load__version new description"
    with test_suite.edit() as editor:
        editor.description(new_description)
        editor.add(test_case_versions[0])

    loaded_test_suite_default = TestSuite.load(test_suite_name)
    loaded_test_suite_v0 = TestSuite.load(test_suite_name, 0)
    loaded_test_suite_v1 = TestSuite.load(test_suite_name, 1)

    assert loaded_test_suite_default == loaded_test_suite_v1

    assert loaded_test_suite_default.version == 1
    assert loaded_test_suite_default.description == new_description

    assert loaded_test_suite_v0.version == 0
    assert loaded_test_suite_v0.description == new_description
    assert loaded_test_suite_v0.test_cases == []

    assert loaded_test_suite_v1.version == 1
    assert loaded_test_suite_v1.description == new_description
    assert loaded_test_suite_v1.test_cases == [test_case_versions[0]]


def test__load__mismatch() -> None:
    test_suite_name = with_test_prefix(f"{__file__} test__load__mismatch test suite")
    DetectionTestSuite(test_suite_name)
    with pytest.raises(WorkflowMismatchError) as exc_info:
        TestSuite.load(test_suite_name)

    exc_info_value = str(exc_info.value)
    assert DetectionTestSuite._workflow.value in exc_info_value
    assert TestSuite._workflow.value in exc_info_value


def test__load__with_version_mismatch() -> None:
    test_suite_name = with_test_prefix(f"{__file__} test__load__with_version_mismatch test suite")
    TestSuite(test_suite_name)
    mismatch_version = 42
    with pytest.raises(NotFoundError) as exc_info:
        TestSuite.load(test_suite_name, mismatch_version)

    exc_info_value = str(exc_info.value)
    assert f"(version {mismatch_version})" in exc_info_value
