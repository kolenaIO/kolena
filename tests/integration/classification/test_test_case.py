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
from typing import List

import pytest

from kolena._api.v1.workflow import WorkflowType
from kolena.classification import TestCase
from kolena.classification import TestImage
from kolena.detection import TestCase as DetectionTestCase
from kolena.errors import NotFoundError
from kolena.errors import WorkflowMismatchError
from tests.integration.detection.helper import assert_test_images_equal
from tests.integration.helper import fake_random_locator
from tests.integration.helper import with_test_prefix


@pytest.fixture(scope="module")
def test_dataset() -> List[TestImage]:
    name = with_test_prefix(f"{__file__}::test_dataset fixture dataset")
    return [
        TestImage(fake_random_locator(), dataset=name),
        TestImage(fake_random_locator(), dataset=name, labels=["car"]),
    ]


def test__create() -> None:
    test_case_name = with_test_prefix(f"{__file__} test__create test case")
    description = f"{test_case_name} (description)"
    test_case = TestCase.create(test_case_name, description)
    assert test_case.version == 0
    assert test_case.name == test_case_name
    assert test_case.description == description
    assert test_case._workflow == WorkflowType.CLASSIFICATION


def test__create__with_images(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__create__with_images test case")
    description = f"{name} (description)"
    images = test_dataset
    test_case = TestCase.create(name, description, images)
    assert test_case.version == 1
    assert test_case._workflow == WorkflowType.CLASSIFICATION
    assert_test_images_equal(test_case.load_images(), images)


def test__load() -> None:
    test_case_name = with_test_prefix(f"{__file__} test__load test case")
    test_case = TestCase(test_case_name)
    loaded_test_case = TestCase.load(test_case_name)
    assert test_case == loaded_test_case


def test__load__with_version() -> None:
    test_case_name = with_test_prefix(f"{__file__} test__load__with_version test case")
    test_case = TestCase(test_case_name)
    new_description = f"{__file__} test__load__version new description"
    with test_case.edit() as editor:
        editor.description(new_description)

    loaded_test_case_default = TestCase.load(test_case_name)
    loaded_test_case_v0 = TestCase.load(test_case_name, 0)
    loaded_test_case_v1 = TestCase.load(test_case_name, 1)

    assert loaded_test_case_default == loaded_test_case_v1

    assert loaded_test_case_default.version == 1
    assert loaded_test_case_default.description == new_description

    assert loaded_test_case_v0.version == 0
    assert loaded_test_case_v0.description == ""

    assert loaded_test_case_v1.version == 1
    assert loaded_test_case_v1.description == new_description


def test__load__mismatch() -> None:
    test_case_name = with_test_prefix(f"{__file__} test__load__mismatch test case")
    DetectionTestCase(test_case_name)
    with pytest.raises(WorkflowMismatchError) as exc_info:
        TestCase.load(test_case_name)

    exc_info_value = str(exc_info.value)
    assert DetectionTestCase._workflow.value in exc_info_value
    assert TestCase._workflow.value in exc_info_value


def test__load__with_version_mismatch() -> None:
    test_case_name = with_test_prefix(f"{__file__} test__load__with_version_mismatch test case")
    TestCase(test_case_name)
    mismatch_version = 42
    with pytest.raises(NotFoundError) as exc_info:
        TestCase.load(test_case_name, mismatch_version)

    exc_info_value = str(exc_info.value)
    assert f"(version {mismatch_version})" in exc_info_value
