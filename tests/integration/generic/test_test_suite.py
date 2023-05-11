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
#
# TODO: these tests are quite similar (but not identical) to other workflow test_test_suite tests. Eventually we
#  should unify the implementations and share all tests to reduce testing burden, reduce behavioral inconsistency, and
#  reduce likelihood of testing holes in a given workflow
#
from typing import List

import pytest

from kolena.detection import TestSuite as DetectionTestSuite
from kolena.errors import NameConflictError
from kolena.errors import WorkflowMismatchError
from tests.integration.generic.dummy import DUMMY_WORKFLOW
from tests.integration.generic.dummy import DummyGroundTruth
from tests.integration.generic.dummy import DummyTestSample
from tests.integration.generic.dummy import TestCase
from tests.integration.generic.dummy import TestSuite
from tests.integration.helper import with_test_prefix


@pytest.fixture(scope="module")
def test_case() -> TestCase:
    name = with_test_prefix(f"{__file__}::test_case fixture test case")
    return TestCase(name, description="test case description")


@pytest.fixture(scope="module")
def test_case_versions(
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> List[TestCase]:
    name = with_test_prefix(f"{__file__}::test_case_versions fixture test case")
    test_case = TestCase(name, description="test case description")
    # load copy such that it is not modified by later edits
    test_case_v0 = TestCase(test_case.name, version=test_case.version)
    with test_case.edit() as editor:
        editor.add(dummy_test_samples[0], dummy_ground_truths[0])
    test_case_v1 = TestCase(test_case.name, version=test_case.version)
    with test_case.edit() as editor:
        editor.add(dummy_test_samples[1], dummy_ground_truths[1])
    test_case_v2 = TestCase(test_case.name, version=test_case.version)
    return [test_case_v0, test_case_v1, test_case_v2]


def test__init() -> None:
    name = with_test_prefix(f"{__file__}::test__init test suite")
    description = "A\n\tlong\ndescription including special characters! 🎉"
    test_suite = TestSuite(name, description=description)
    assert test_suite.workflow == DUMMY_WORKFLOW
    assert test_suite.name == name
    assert test_suite.version == 0
    assert test_suite.description == description
    assert test_suite.test_cases == []

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
    test_suite = TestSuite(name, description="initial description")

    test_suite0 = TestSuite(name, version=test_suite.version)
    assert test_suite == test_suite0

    with pytest.raises(NameConflictError):  # TODO: this should be a NotFoundError when version is specified
        TestSuite(name, version=123)

    with test_suite.edit() as editor:
        new_description = "new description"
        editor.description(new_description)
        editor.add(test_case_versions[0])

    assert test_suite.version == test_suite0.version  # editing description does not bump version
    assert test_suite.description == new_description
    assert test_suite == TestSuite(name, version=test_suite.version)
    assert test_suite == TestSuite(name)

    test_suite0_reloaded = TestSuite(name, version=test_suite0.version)
    assert test_suite0.test_cases == test_suite0_reloaded.test_cases
    assert test_suite0_reloaded.description == new_description


def test__load__mismatching_workflows() -> None:
    name = with_test_prefix(f"{__file__}::test__load__mismatching_workflows")
    DetectionTestSuite(name)
    with pytest.raises(WorkflowMismatchError):
        TestSuite(name)


def test__edit(test_case_versions: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit test suite")
    description = "test__edit test suite description"
    test_suite = TestSuite(name, description=description, test_cases=[test_case_versions[0]])
    assert test_suite.name == name
    assert test_suite.version == 1
    assert test_suite.description == description
    assert test_suite.test_cases == [test_case_versions[0]]
    assert all(tc.workflow == test_suite.workflow for tc in test_suite.test_cases)

    test_case = TestCase(f"{__file__}::test__edit test suite test case")
    with test_suite.edit() as editor:
        editor.add(test_case)
    assert test_suite.version == 2
    assert test_suite.test_cases == [test_case_versions[0], test_case]  # note that ordering matters


def test__edit__no_op(test_case_versions: List[TestCase]) -> None:
    test_suite = TestSuite(with_test_prefix(f"{__file__}::test__edit__no_op test suite"))
    with test_suite.edit():
        ...
    assert test_suite.version == 0

    with test_suite.edit() as editor:  # not in suite
        editor.remove(test_case_versions[0])
    assert test_suite.version == 0

    with test_suite.edit() as editor:
        editor.add(test_case_versions[0])
        editor.remove(test_case_versions[0])
    assert test_suite.version == 0


def test__edit__same_name_test_case(test_case_versions: List[TestCase]) -> None:
    test_suite = TestSuite(with_test_prefix(f"{__file__}::test__edit__same_name_test_case test suite"))
    with test_suite.edit() as editor:
        editor.add(test_case_versions[0])

    # latest added should be only version in suite, even if a version of that test case is already present
    with test_suite.edit() as editor:
        for test_case in test_case_versions[1:]:
            editor.add(test_case)
    assert test_suite.version == 2
    assert test_suite.test_cases == [test_case_versions[-1]]


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


def test__load_test_samples(
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    test_case_1 = TestCase(with_test_prefix(f"{__file__}::test__load_test_samples test case 1"))
    with test_case_1.edit() as editor:
        editor.add(dummy_test_samples[0], dummy_ground_truths[0])
        editor.add(dummy_test_samples[1], dummy_ground_truths[1])

    test_case_2 = TestCase(with_test_prefix(f"{__file__}::test__load_test_samples test case 2"))
    with test_case_2.edit() as editor:
        editor.add(dummy_test_samples[2], dummy_ground_truths[2])
        editor.add(dummy_test_samples[1], dummy_ground_truths[1])

    test_suite = TestSuite(
        with_test_prefix(f"{__file__}::test__load_test_samples test suite"),
        test_cases=[
            test_case_1,
            test_case_2,
        ],
    )
    test_case_test_samples = test_suite.load_test_samples()

    assert [(test_case, sorted(test_samples)) for test_case, test_samples in test_case_test_samples] == [
        (test_case_1, sorted([dummy_test_samples[0], dummy_test_samples[1]])),
        (test_case_2, sorted([dummy_test_samples[2], dummy_test_samples[1]])),
    ]
