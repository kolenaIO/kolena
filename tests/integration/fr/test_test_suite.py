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

from kolena.errors import NameConflictError
from kolena.errors import NotFoundError
from kolena.errors import RemoteError
from kolena.fr import TestCase
from kolena.fr import TestSuite
from kolena.fr.datatypes import TestCaseRecord
from tests.integration.helper import with_test_prefix


@pytest.fixture(scope="module")
def single_test_case(test_samples: List[TestCaseRecord]) -> TestCase:
    name = with_test_prefix(f"{__file__}::test_case fixture test case")
    return TestCase(name, description="test case description", test_samples=test_samples)


@pytest.fixture(scope="module")
def multi_version_test_case(test_samples: List[TestCaseRecord]) -> List[TestCase]:
    name = with_test_prefix(f"{__file__}::test_case_versions fixture test case")
    test_case = TestCase(name, description="test case description")
    # load copy such that it is not modified by later edits
    test_case_v0 = TestCase(test_case.name, version=test_case.version)
    with test_case.edit() as editor:
        editor.add(*test_samples[0])
    test_case_v1 = TestCase(test_case.name, version=test_case.version)
    with test_case.edit() as editor:
        editor.add(*test_samples[1])
    test_case_v2 = TestCase(test_case.name, version=test_case.version)
    return [test_case_v0, test_case_v1, test_case_v2]


def test__init() -> None:
    name = with_test_prefix(f"{__file__}::test__init test suite")
    description = "A\n\tlong\ndescription including special characters! 🎉"
    test_suite = TestSuite(name, description=description)
    assert test_suite.name == name
    assert test_suite.version == 0
    assert test_suite.description == description
    assert test_suite.baseline_test_cases == []
    assert test_suite.non_baseline_test_cases == []
    assert test_suite.baseline_image_count == 0
    assert test_suite.baseline_pair_count_genuine == 0
    assert test_suite.baseline_pair_count_imposter == 0

    test_suite2 = TestSuite(name)
    assert test_suite == test_suite2

    test_suite3 = TestSuite(name, description="different description should be ignored")
    assert test_suite == test_suite3


def test__init__with_test_cases(single_test_case: TestCase) -> None:
    name = with_test_prefix(f"{__file__}::test__init__with_test_cases test suite")
    description = "A\n\tlong\ndescription including special characters! 🎉"
    test_suite = TestSuite(name, description=description, baseline_test_cases=[single_test_case])
    assert test_suite.name == name
    assert test_suite.version == 1
    assert test_suite.description == description
    assert test_suite.baseline_test_cases == [single_test_case]
    assert test_suite.non_baseline_test_cases == []
    assert test_suite.baseline_image_count == 3
    assert test_suite.baseline_pair_count_genuine == 3
    assert test_suite.baseline_pair_count_imposter == 6


def test__init__no_baseline_error(single_test_case: TestCase) -> None:
    name = with_test_prefix(f"{__file__}::test__init__no_baseline_error test suite")
    description = "A\n\tlong\ndescription including special characters! 🎉"
    expected_error_msg = "to a state without any baseline test cases"

    with pytest.raises(RemoteError) as exc_info:
        TestSuite(name, description=description, non_baseline_test_cases=[single_test_case])
    exc_info_value = str(exc_info.value)
    assert expected_error_msg in exc_info_value

    with pytest.raises(RemoteError) as exc_info:
        TestSuite(name, description=description, non_baseline_test_cases=[single_test_case], reset=True)
    exc_info_value = str(exc_info.value)
    assert expected_error_msg in exc_info_value


def test__init__with_version(single_test_case: TestCase) -> None:
    name = with_test_prefix(f"{__file__}::test__init__with_version test suite")
    description = "test suite description"
    test_suite = TestSuite(name, description=description, baseline_test_cases=[single_test_case])

    test_suite0 = TestSuite(name, version=test_suite.version)
    assert test_suite == test_suite0

    with pytest.raises(NameConflictError):  # TODO: should raise NotFoundError when version is specified
        TestSuite(name, version=123)

    new_description = "new description"
    with test_suite.edit() as editor:  # description-only edit does not bump version
        editor.description(new_description)

    assert test_suite.description == new_description
    assert test_suite.version == test_suite0.version
    assert test_suite == TestSuite(name, version=test_suite.version)
    assert test_suite == TestSuite(name)


def test__init__reset(single_test_case: TestCase, multi_version_test_case: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__reset test suite")
    description = f"{name} (description)"
    TestSuite(
        name,
        description=description,
        baseline_test_cases=[single_test_case],
        non_baseline_test_cases=[multi_version_test_case[0]],
    )

    new_test_cases = [single_test_case]
    test_suite = TestSuite(name, baseline_test_cases=new_test_cases, reset=True)
    assert test_suite.version == 2
    assert test_suite.description == description  # not updated or cleared
    assert test_suite.baseline_test_cases == new_test_cases
    assert test_suite.non_baseline_test_cases == []

    new_test_cases = [multi_version_test_case[1]]
    test_suite = TestSuite(name, baseline_test_cases=new_test_cases, reset=True)
    assert test_suite.version == 3
    assert test_suite.description == description  # not updated or cleared
    assert test_suite.baseline_test_cases == new_test_cases
    assert test_suite.non_baseline_test_cases == []


def test__load(fr_test_suites: List[TestSuite], fr_test_cases: List[TestCase]) -> None:
    test_suite = fr_test_suites[0]
    test_suite_updated = fr_test_suites[1]

    loaded_test_suite = TestSuite.load(test_suite.name)
    assert loaded_test_suite._id == test_suite_updated._id
    assert loaded_test_suite.name == test_suite.name
    assert loaded_test_suite.description == test_suite_updated.description
    assert loaded_test_suite.version == test_suite_updated.version

    assert loaded_test_suite.baseline_test_cases == [fr_test_cases[1].data]
    assert loaded_test_suite.non_baseline_test_cases == [fr_test_cases[2].data]


def test__load__with_version(fr_test_suites: List[TestSuite], fr_test_cases: List[TestCase]) -> None:
    # the test suite is an older version
    test_suite = fr_test_suites[0]
    loaded_test_suite = TestSuite.load(test_suite.name, version=test_suite.version)

    assert loaded_test_suite._id == test_suite._id
    assert loaded_test_suite.name == test_suite.name
    assert loaded_test_suite.version == test_suite.version
    # note: not checking description as this is independent of version

    assert [tc.data for tc in loaded_test_suite.baseline_test_cases] == [fr_test_cases[0].data]
    assert [tc.data for tc in loaded_test_suite.non_baseline_test_cases] == [fr_test_cases[2].data]


def test__load__absent(fr_test_suites: List[TestSuite]) -> None:
    with pytest.raises(NotFoundError):
        TestSuite.load("name of a test suite that does not exist")

    with pytest.raises(NotFoundError):
        # names should be case-sensitive
        TestSuite.load(fr_test_suites[0].name.lower())


def test__create() -> None:
    name = with_test_prefix(f"{__file__}::test__create test suite")
    description = "\tSome test suite description\nspanning\nmultiple lines."
    test_suite = TestSuite.create(name, description=description)
    assert test_suite.name == name
    assert test_suite.version == 0
    assert test_suite.description == description

    test_suite = TestSuite.create(with_test_prefix(f"{__file__}::test_create test suite 2"))
    assert test_suite.description == ""


def test__edit(fr_test_cases: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit test suite")
    test_suite = TestSuite.create(name)
    test_cases = fr_test_cases

    new_description = "some new test suite description"
    with test_suite.edit() as editor:
        editor.description(new_description)
        editor.add(TestCase.load(test_cases[0].name), is_baseline=True)
        editor.add(TestCase.load(test_cases[2].name))
        editor.remove(TestCase.load(test_cases[2].name))

    assert test_suite.version == 1
    assert test_suite.description == new_description
    all_test_cases = test_suite.baseline_test_cases + test_suite.non_baseline_test_cases
    actual_names = sorted(tc.name for tc in all_test_cases)
    expected_names = sorted([test_cases[0].name])
    assert actual_names == expected_names

    # removal of test case not in suite raises exception and kills entire transaction
    with pytest.raises(KeyError):
        with test_suite.edit() as editor:
            editor.remove(TestCase.load(test_cases[2].name))

    assert actual_names == expected_names


# Note: editor.merge is deprecated
def test__edit__merge(fr_test_cases: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__merge test suite")
    test_suite = TestSuite.create(name)
    test_cases = fr_test_cases

    with test_suite.edit() as editor:
        editor.add(TestCase.load(test_cases[0].name), is_baseline=True)

    # merge of an existing test case replaces the previous and propagates its baseline status
    with test_suite.edit() as editor:
        editor.merge(TestCase.load(test_cases[1].name))
        editor.add(TestCase.load(test_cases[2].name))

    assert test_suite.version == 2
    all_test_cases = test_suite.baseline_test_cases + test_suite.non_baseline_test_cases
    actual_names = sorted(tc.name for tc in all_test_cases)
    expected_names = sorted([tc.name for tc in test_cases[1:3]])
    assert actual_names == expected_names
    # expect updated test case to still be considered the baseline
    assert test_suite.baseline_test_cases[0].version == test_cases[1].version


def test__edit__no_op() -> None:
    name = with_test_prefix(f"{__file__}::test__edit__no_op test suite")
    test_suite = TestSuite.create(name)
    version = test_suite.version
    # no-op editor contexts do not bump version
    with test_suite.edit():
        ...
    assert test_suite.version == version


def test__edit__same_name(fr_test_cases: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__same_name test suite")
    test_suite = TestSuite.create(name)
    # test_case_1 is updated version of test_case_0
    test_case_0 = fr_test_cases[0]
    test_case_1 = fr_test_cases[1]

    with test_suite.edit() as editor:
        editor.add(test_case_0, is_baseline=True)
    assert test_suite.baseline_test_cases == [test_case_0]

    # multiple versions of the same test will be updated and keep the baseline status
    with test_suite.edit() as editor:
        editor.add(test_case_0, is_baseline=True)
        editor.add(test_case_1)
    assert test_suite.baseline_test_cases == [test_case_1]

    with test_suite.edit() as editor:
        editor.add(test_case_1)
    assert test_suite.baseline_test_cases == [test_case_1]


def test__edit__empty(fr_test_cases: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__empty test suite")
    test_suite = TestSuite.create(name)
    test_case = fr_test_cases[0]

    with test_suite.edit() as editor:
        editor.add(test_case, is_baseline=True)

    # refuse to leave a test suite in an empty state
    with pytest.raises(RemoteError):
        with test_suite.edit() as editor:
            editor.remove(test_case)


def test__edit__baseline_counts(fr_test_cases: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__baseline_counts test suite")
    test_suite = TestSuite.create(name)
    assert test_suite.baseline_image_count == 0
    assert test_suite.baseline_pair_count_genuine == 0
    assert test_suite.baseline_pair_count_imposter == 0

    test_case_record = fr_test_cases[0]
    test_case = TestCase.load(test_case_record.name, version=test_case_record.version)
    with test_suite.edit() as editor:
        editor.add(test_case, is_baseline=True)

    assert test_suite.baseline_image_count == 3
    assert test_suite.baseline_pair_count_genuine == 3
    assert test_suite.baseline_pair_count_imposter == 1

    # this test case has overlapping images and pairs with the previously added test case
    test_case_record = fr_test_cases[2]
    test_case = TestCase.load(test_case_record.name, version=test_case_record.version)
    with test_suite.edit() as editor:
        editor.add(test_case, is_baseline=True)

    # assert images and pairs are properly deduped
    assert test_suite.baseline_image_count == 4
    assert test_suite.baseline_pair_count_genuine == 3
    assert test_suite.baseline_pair_count_imposter == 2


def test__edit__no_baseline_error(single_test_case: TestCase) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__no_baseline_error test suite")
    test_suite = TestSuite(name)
    expected_error_msg = "to a state without any baseline test cases"

    with pytest.raises(RemoteError) as exc_info:
        with test_suite.edit() as editor:
            editor.description("some description")
    exc_info_value = str(exc_info.value)
    assert expected_error_msg in exc_info_value

    with pytest.raises(RemoteError) as exc_info:
        with test_suite.edit() as editor:
            editor.merge(single_test_case, False)
    exc_info_value = str(exc_info.value)
    assert expected_error_msg in exc_info_value

    with pytest.raises(RemoteError) as exc_info:
        with test_suite.edit() as editor:
            editor.description("some description")
            editor.add(single_test_case, False)
    exc_info_value = str(exc_info.value)
    assert expected_error_msg in exc_info_value

    with pytest.raises(RemoteError) as exc_info:
        with test_suite.edit() as editor:
            editor.description("some description")
            editor.add(single_test_case, True)
            editor.remove(single_test_case)
    exc_info_value = str(exc_info.value)
    assert expected_error_msg in exc_info_value

    with test_suite.edit() as editor:
        editor.add(single_test_case, True)

    with pytest.raises(RemoteError) as exc_info:
        with test_suite.edit() as editor:
            editor.remove(single_test_case)
    exc_info_value = str(exc_info.value)
    assert expected_error_msg in exc_info_value


def test__edit__reset(single_test_case: TestCase, multi_version_test_case: List[TestCase]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__reset test suite")
    test_suite = TestSuite(
        name,
        baseline_test_cases=[single_test_case],
        non_baseline_test_cases=[multi_version_test_case[2]],
    )

    with test_suite.edit(reset=True) as editor:
        editor.add(multi_version_test_case[1], True)
    assert test_suite.version == 2
    assert test_suite.description == ""
    assert test_suite.baseline_test_cases == [multi_version_test_case[1]]

    new_description = "new description"
    with test_suite.edit(reset=True) as editor:  # no change to test suite contents does not bump version
        editor.description(new_description)
        editor.add(multi_version_test_case[1], True)
    assert test_suite.version == 2
    assert test_suite.description == new_description  # updated without version bump
    assert test_suite.baseline_test_cases == [multi_version_test_case[1]]

    with test_suite.edit(reset=True) as editor:
        editor.add(multi_version_test_case[2], True)
    assert test_suite.version == 3
    assert test_suite.description == new_description  # not updated or cleared
    assert test_suite.baseline_test_cases == [multi_version_test_case[2]]
