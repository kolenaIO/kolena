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

from kolena._api.v1.workflow import WorkflowType
from kolena.classification import TestCase as ClassificationTestCase
from kolena.classification import TestSuite as ClassificationTestSuite
from kolena.detection import TestCase
from kolena.detection import TestImage
from kolena.detection import TestSuite
from kolena.detection.ground_truth import ClassificationLabel
from kolena.errors import NameConflictError
from kolena.errors import NotFoundError
from kolena.errors import WorkflowMismatchError
from tests.integration.helper import fake_random_locator
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
        TestImage(fake_random_locator(), dataset=name, ground_truths=[ClassificationLabel("car")]),
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


def test__init() -> None:
    name = with_test_prefix(f"{__file__}::test__init test suite")
    description = "A\n\tlong\ndescription including special characters! 🎉"
    test_suite = TestSuite(name, description=description)
    assert test_suite.name == name
    assert test_suite.version == 0
    assert test_suite.description == description
    assert test_suite.test_cases == []
    assert test_suite._workflow == WorkflowType.DETECTION

    test_suite2 = TestSuite(name)
    assert test_suite == test_suite2

    test_suite3 = TestSuite(name, description="different description should be ignored")
    assert test_suite == test_suite3


def test__init__reset(test_case: TestCase, test_case_versions: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__reset test suite")
    description = f"{name} (description)"
    TestSuite(name, description=description, test_cases=[test_case, test_case_versions[0]])

    new_test_cases = [test_case_versions[1]]
    test_suite = TestSuite(name, test_cases=new_test_cases, reset=True)
    assert test_suite.version == 2
    assert test_suite.description == description  # not updated or cleared
    assert test_suite.test_cases == new_test_cases


def test__init__with_version(test_case_versions: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__with_version test suite")
    description = "test suite description"
    test_suite = TestSuite(name, description=description)

    test_suite0 = TestSuite(name, version=test_suite.version)
    assert test_suite == test_suite0

    with pytest.raises(NameConflictError):
        TestSuite(name, version=123)

    with test_suite.edit() as editor:
        new_description = "new description"
        editor.description(new_description)
        editor.add(test_case_versions[0])

    assert test_suite.description == new_description
    assert test_suite == TestSuite(name, version=test_suite.version)
    assert test_suite == TestSuite(name)
    assert test_suite.test_cases == [test_case_versions[0]]

    test_suite0_reloaded = TestSuite(name, version=test_suite0.version)
    assert test_suite0.test_cases == test_suite0_reloaded.test_cases
    assert test_suite0_reloaded.description == new_description
    assert test_suite0_reloaded.test_cases == []


def test__edit(test_case: TestCase) -> None:
    name = with_test_prefix(f"{__file__}::test__edit test suite")
    description = "test__edit test suite description"
    test_suite = TestSuite(name, description=description)
    with test_suite.edit() as editor:
        editor.add(test_case)
    assert test_suite.name == name
    assert test_suite.version == 1
    assert test_suite.description == description
    assert test_suite.test_cases == [test_case]
    assert test_suite._workflow == WorkflowType.DETECTION
    assert all(tc._workflow == WorkflowType.DETECTION for tc in test_suite.test_cases)

    test_case0 = TestCase(with_test_prefix(f"{__file__}::test__edit test suite test case"))
    with test_suite.edit() as editor:
        editor.add(test_case0)
    assert test_suite.version == 2
    assert test_suite.test_cases == [test_case, test_case0]  # note that ordering matters


def test__edit__no_op(test_case: TestCase) -> None:
    test_suite = TestSuite(with_test_prefix(f"{__file__}::test__edit__no_op test suite"))
    with test_suite.edit():
        ...
    assert test_suite.version == 0

    with test_suite.edit() as editor:
        editor.add(test_case)
        editor.remove(test_case)
    assert test_suite.version == 0
    assert test_suite.test_cases == []


def test__edit__idempotent(test_case: TestCase, test_case_versions: List[TestCase]) -> None:
    test_cases = [test_case, test_case_versions[0]]
    test_suite = TestSuite(with_test_prefix(f"{__file__}::test__edit__no_op test suite"), test_cases=test_cases)
    assert test_suite.version == 1

    # adding the same test cases in the same order doesn't edit the suite, no-op
    with test_suite.edit() as editor:
        for tc in test_cases:
            editor.add(tc)
    assert test_suite.version == 1
    assert test_suite.test_cases == test_cases


def test__edit__same_name_test_case(test_case_versions: List[TestCase]) -> None:
    test_suite = TestSuite(with_test_prefix(f"{__file__}::test__edit__same_name_test_case test suite"))
    with test_suite.edit() as editor:
        editor.add(test_case_versions[2])

    # a version is already in the test suite, 'add' should replace existing version
    for test_case in test_case_versions:
        with test_suite.edit() as editor:
            editor.add(test_case)
        assert test_suite.test_cases[0].version == test_case.version


def test__edit__add(test_case: TestCase, test_case_versions: List[TestCase]) -> None:
    test_suite = TestSuite(with_test_prefix(f"{__file__}::test__edit__add test suite"))
    with test_suite.edit() as editor:
        editor.add(test_case)
        editor.add(test_case_versions[0])  # same as add when the test case isn't already present
    assert test_suite.version == 1
    assert test_suite.test_cases == [test_case, test_case_versions[0]]
    previous_test_cases = test_suite.test_cases

    with test_suite.edit() as editor:
        editor.add(test_case)  # no-op
        editor.add(test_case_versions[1])  # should replace the existing test_case_version
        editor.add(test_case_versions[2])  # should replace the test_case_version added in the above line
    assert test_suite.version == 2
    assert test_suite.test_cases == [test_case, test_case_versions[2]]
    assert test_suite.test_cases != previous_test_cases


def test__edit__add_mismatch_workflow() -> None:
    test_suite_name = with_test_prefix(f"{__file__}::test__edit__add_mismatch_workflow test suite")
    test_suite = TestSuite(test_suite_name)
    classification_test_case = ClassificationTestCase(f"{test_suite_name}::classification_test_case")
    with test_suite.edit() as editor:
        with pytest.raises(ValueError) as exc_info:
            editor.add(classification_test_case)
        exc_info_value = str(exc_info.value)
        assert WorkflowType.CLASSIFICATION.value in exc_info_value
        assert WorkflowType.DETECTION.value in exc_info_value


def test__edit__merge(test_case: TestCase, test_case_versions: List[TestCase]) -> None:
    test_suite = TestSuite(with_test_prefix(f"{__file__}::test__edit__merge test suite"))
    with test_suite.edit() as editor:
        editor.add(test_case)
        editor.merge(test_case_versions[0])  # same as add when the test case isn't already present
    assert test_suite.version == 1
    assert test_suite.test_cases == [test_case, test_case_versions[0]]
    previous_test_cases = test_suite.test_cases

    with test_suite.edit() as editor:
        editor.merge(test_case)  # no-op
        editor.merge(test_case_versions[1])  # should replace the existing test_case_version
        editor.merge(test_case_versions[2])  # should replace the test_case_version merged in the above line
    assert test_suite.version == 2
    assert test_suite.test_cases == [test_case, test_case_versions[2]]
    assert test_suite.test_cases != previous_test_cases


def test__edit__reset(test_case: TestCase, test_case_versions: List[TestCase]) -> None:
    test_suite = TestSuite(
        with_test_prefix(f"{__file__}::test__edit__reset test suite"),
        test_cases=[
            test_case,
            test_case_versions[0],
        ],
    )
    new_description = "new description"

    with test_suite.edit(reset=True) as editor:
        editor.description(new_description)
        editor.add(test_case_versions[1])
    assert test_suite.version == 2
    assert test_suite.description == new_description
    assert test_suite.test_cases == [test_case_versions[1]]

    with test_suite.edit(reset=True) as editor:  # no-op
        editor.add(test_case_versions[1])
    assert test_suite.version == 2
    assert test_suite.description == new_description
    assert test_suite.test_cases == [test_case_versions[1]]


def test__create() -> None:
    test_suite_name = with_test_prefix(f"{__file__} test__create test suite")
    description = "A\n\tlong\ndescription including special characters! 🎉"
    test_suite = TestSuite.create(test_suite_name, description=description)
    assert test_suite.name == test_suite_name
    assert test_suite.version == 0
    assert test_suite.description == description
    assert test_suite.test_cases == []
    assert test_suite._workflow == WorkflowType.DETECTION


def test__create__with_test_cases(test_case: TestCase, test_case_versions: List[TestCase]) -> None:
    test_suite_name = with_test_prefix(f"{__file__} test__create__with_test_cases test suite")
    description = "A\n\tlong\ndescription including special characters! 🎉"
    test_cases = [test_case, test_case_versions[0]]
    test_suite = TestSuite.create(test_suite_name, description=description, test_cases=test_cases)
    assert test_suite.name == test_suite_name
    assert test_suite.version == 1
    assert test_suite.description == description
    assert test_suite.test_cases == test_cases
    assert test_suite._workflow == WorkflowType.DETECTION


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
    assert loaded_test_suite_default.test_cases == [test_case_versions[0]]

    assert loaded_test_suite_v0.version == 0
    assert loaded_test_suite_v0.description == new_description
    assert loaded_test_suite_v0.test_cases == []

    assert loaded_test_suite_v1.version == 1
    assert loaded_test_suite_v1.description == new_description
    assert loaded_test_suite_v1.test_cases == [test_case_versions[0]]


def test__load__mismatch() -> None:
    test_suite_name = with_test_prefix(f"{__file__} test__load__mismatch test suite")
    ClassificationTestSuite(test_suite_name)
    with pytest.raises(WorkflowMismatchError) as exc_info:
        TestSuite.load(test_suite_name)

    exc_info_value = str(exc_info.value)
    assert ClassificationTestSuite._workflow.value in exc_info_value
    assert TestSuite._workflow.value in exc_info_value


def test__load__with_version_mismatch() -> None:
    test_suite_name = with_test_prefix(f"{__file__} test__load__with_version_mismatch test suite")
    TestSuite(test_suite_name)
    mismatch_version = 42
    with pytest.raises(NotFoundError) as exc_info:
        TestSuite.load(test_suite_name, mismatch_version)

    exc_info_value = str(exc_info.value)
    assert f"{mismatch_version}" in exc_info_value


def test__empty_test_suite_name() -> None:
    test_case_1 = TestCase(with_test_prefix(f"{__file__}::test__empty_test_suite_name test case 1"))
    with pytest.raises(ValueError):
        TestSuite("", test_cases=[test_case_1])
