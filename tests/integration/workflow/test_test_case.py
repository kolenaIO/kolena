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
from typing import Optional

import pytest

from tests.integration.helper import assert_sorted_list_equal
from tests.integration.helper import with_test_prefix
from tests.integration.workflow.dummy import DUMMY_WORKFLOW
from tests.integration.workflow.dummy import DummyGroundTruth
from tests.integration.workflow.dummy import DummyTestSample
from tests.integration.workflow.dummy import TestCase


def assert_test_case(test_case: TestCase, name: str, version: int, description: Optional[str] = None) -> None:
    assert test_case.workflow == DUMMY_WORKFLOW
    assert test_case.name == name
    assert test_case.version == version

    if description:
        assert test_case.description == description


def test__init(
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    name = with_test_prefix(f"{__file__}::test__init test case")
    assert_test_case(TestCase(name), name, 0)  # should create
    assert_test_case(TestCase(name), name, 0)  # should load

    all_test_samples = list(zip(dummy_test_samples, dummy_ground_truths))
    test_case = TestCase(name, test_samples=all_test_samples)
    assert test_case.version == 1
    assert_sorted_list_equal(test_case.load_test_samples(), all_test_samples)


def test__init__reset(
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    name = with_test_prefix(f"{__file__}::test__init__reset test case")
    description = f"{name} (description)"
    TestCase(name, description=description, test_samples=list(zip(dummy_test_samples, dummy_ground_truths)))

    new_test_samples = list(zip(dummy_test_samples[:2][::-1], dummy_ground_truths[:2][::1]))
    test_case = TestCase(name, test_samples=new_test_samples, reset=True)
    assert test_case.version == 2
    assert test_case.description == description  # not updated or cleared
    assert sorted(test_case.load_test_samples()) == sorted(new_test_samples)


def test__create() -> None:
    name = with_test_prefix(f"{__file__}::test__create test case")
    assert_test_case(TestCase.create(name), name, 0)

    with pytest.raises(Exception):  # TODO: better error?
        TestCase.create(name)


def test__load() -> None:
    name = with_test_prefix(f"{__file__}::test__load test case")

    with pytest.raises(Exception):  # TODO: better error?
        TestCase.load(name)

    TestCase.create(name)
    assert_test_case(TestCase.load(name), name, 0)


def test__edit(
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    test_case = TestCase(with_test_prefix(f"{__file__}::test__edit test case"))
    assert test_case.version == 0

    description = "new description"
    all_samples = list(zip(dummy_test_samples, dummy_ground_truths))
    with test_case.edit() as editor:
        editor.description(description)
        for test_sample, ground_truth in all_samples:
            editor.add(test_sample, ground_truth)

    assert test_case.version == 1
    assert test_case.description == description
    assert sorted(test_case.load_test_samples()) == sorted(all_samples)


def test__edit__reset(
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    test_case = TestCase(
        with_test_prefix(f"{__file__}::test__edit__reset test case"),
        test_samples=[(dummy_test_samples[0], dummy_ground_truths[0]), (dummy_test_samples[1], dummy_ground_truths[1])],
    )

    added = [
        (dummy_test_samples[2], dummy_ground_truths[2]),
        (dummy_test_samples[1], dummy_ground_truths[3]),  # re-add sample that was previously present
    ]
    with test_case.edit(reset=True) as editor:
        for tc, gt in added:
            editor.add(tc, gt)

    assert test_case.version == 2
    assert sorted(test_case.load_test_samples()) == sorted(added)


def test__edit__replace(
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    test_case = TestCase(with_test_prefix(f"{__file__}::test__edit__replace test case"))

    # one pass, first is shadowed
    with test_case.edit() as editor:
        editor.add(dummy_test_samples[0], dummy_ground_truths[0])
        editor.add(dummy_test_samples[0], dummy_ground_truths[1])

    assert test_case.load_test_samples() == [(dummy_test_samples[0], dummy_ground_truths[1])]

    # two passes
    with test_case.edit() as editor:
        editor.add(dummy_test_samples[0], dummy_ground_truths[2])

    assert sorted(test_case.load_test_samples()) == sorted([(dummy_test_samples[0], dummy_ground_truths[2])])


def test__edit__remove(
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    test_case = TestCase(
        with_test_prefix(f"{__file__}::test__edit__remove test case"),
        test_samples=[
            (dummy_test_samples[0], dummy_ground_truths[0]),
            (dummy_test_samples[1], dummy_ground_truths[1]),
        ],
    )

    with test_case.edit() as editor:
        editor.remove(dummy_test_samples[0])
        editor.add(dummy_test_samples[2], dummy_ground_truths[0])

    assert test_case.version == 2
    assert test_case.load_test_samples() == [
        (dummy_test_samples[1], dummy_ground_truths[1]),
        (dummy_test_samples[2], dummy_ground_truths[0]),
    ]

    with test_case.edit() as editor:
        editor.remove(dummy_test_samples[0])  # removing sample not in case is fine
        editor.remove(dummy_test_samples[1])
        editor.remove(dummy_test_samples[1])  # removing sample twice is fine
        editor.add(dummy_test_samples[1], dummy_ground_truths[2])  # add sample back in

    assert test_case.version == 3
    assert test_case.load_test_samples() == [
        (dummy_test_samples[2], dummy_ground_truths[0]),
        (dummy_test_samples[1], dummy_ground_truths[2]),
    ]


def test__edit__remove_only(
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> None:
    ts0, *ts_rest = dummy_test_samples
    gt0, *gt_rest = dummy_ground_truths
    rest = list(zip(ts_rest, gt_rest))
    test_case = TestCase(
        with_test_prefix(f"{__file__}::test__edit__remove_only test case"),
        test_samples=[(ts0, gt0), *rest],
    )

    with test_case.edit() as editor:
        editor.remove(ts0)

    assert test_case.version == 2
    assert sorted(test_case.load_test_samples()) == sorted(rest)


def test__init__validate_name() -> None:
    with pytest.raises(ValueError):
        TestCase("")
