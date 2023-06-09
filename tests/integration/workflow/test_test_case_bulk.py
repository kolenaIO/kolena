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
from typing import Any
from typing import Dict
from typing import List
from typing import Tuple

from tests.integration.helper import with_test_prefix
from tests.integration.workflow.conftest import dummy_ground_truth
from tests.integration.workflow.conftest import dummy_test_sample
from tests.integration.workflow.dummy import DummyGroundTruth
from tests.integration.workflow.dummy import DummyTestSample
from tests.integration.workflow.dummy import TestCase
from tests.integration.workflow.test_test_case import assert_test_case


def verify_test_samples(test_case: TestCase, expected_samples: List[Tuple[DummyTestSample, DummyGroundTruth]]) -> None:
    loaded_test_samples = test_case.load_test_samples()
    assert sorted(loaded_test_samples) == sorted(expected_samples)


def get_test_cass(response: Dict[str, Any]) -> List[TestCase]:
    return [tc["data"] for tc in response["test_cases"]]


def test__init_many() -> None:
    name = with_test_prefix(f"{__file__}::test__init_many")
    test_samples = [(dummy_test_sample(i, "test__init_many"), dummy_ground_truth(i)) for i in range(10)]
    test_case_edits = [
        (f"{name} 1", test_samples[:3]),
        (f"{name} 2", test_samples[4:9]),
        (f"{name} 3", []),
    ]

    test_cases = TestCase.init_many(test_case_edits)

    assert_test_case(test_cases[0], f"{name} 1", 1, "")
    assert_test_case(test_cases[1], f"{name} 2", 1, "")
    assert_test_case(test_cases[2], f"{name} 3", 0, "")

    verify_test_samples(test_cases[0], test_samples[:3])
    verify_test_samples(test_cases[1], test_samples[4:9])
    verify_test_samples(test_cases[2], [])


def test__init_many__reset() -> None:
    name = with_test_prefix(f"{__file__}::test__init_many__reset")
    test_samples = [(dummy_test_sample(i, "test__init_many__reset"), dummy_ground_truth(i)) for i in range(10)]
    test_case_edits = [
        (f"{name} 1", test_samples[:3]),
        (f"{name} 2", test_samples[4:9]),
        (f"{name} 3", []),
    ]

    test_cases = TestCase.init_many(test_case_edits, reset=True)

    assert_test_case(test_cases[0], f"{name} 1", 1, "")
    assert_test_case(test_cases[1], f"{name} 2", 1, "")
    assert_test_case(test_cases[2], f"{name} 3", 0, "")

    verify_test_samples(test_cases[0], test_samples[:3])
    verify_test_samples(test_cases[1], test_samples[4:9])
    verify_test_samples(test_cases[2], [])


def test__init_many__existing() -> None:
    name = with_test_prefix(f"{__file__}::test__init_many__existing")
    test_samples = [(dummy_test_sample(i, "test__init_many__existing"), dummy_ground_truth(i)) for i in range(10)]
    test_cases = [
        TestCase(f"{name} 1", description="desc 1"),
        TestCase(f"{name} 2", description="desc 2", test_samples=test_samples[:3]),
        TestCase(f"{name} 3", description="desc 3", test_samples=test_samples[3:6]),
    ]
    loaded_test_cases = TestCase.init_many(
        [
            (test_cases[0].name, []),
            (test_cases[1].name, test_samples[3:6]),
            (test_cases[2].name, test_samples[6:9]),
        ],
    )

    assert_test_case(loaded_test_cases[0], test_cases[0].name, 0, "desc 1")
    assert_test_case(loaded_test_cases[1], test_cases[1].name, 1, "desc 2")
    assert_test_case(loaded_test_cases[2], test_cases[2].name, 1, "desc 3")

    verify_test_samples(loaded_test_cases[0], [])
    verify_test_samples(loaded_test_cases[1], test_samples[:3])
    verify_test_samples(loaded_test_cases[2], test_samples[3:6])


def test__init_many__existing_reset() -> None:
    name = with_test_prefix(f"{__file__}::test__init_many__existing_reset")
    test_samples = [(dummy_test_sample(i, "test__init_many__existing_reset"), dummy_ground_truth(i)) for i in range(10)]
    test_cases = [
        TestCase(f"{name} 1", description="desc 1"),
        TestCase(f"{name} 2", description="desc 2", test_samples=test_samples[:3]),
        TestCase(f"{name} 3", description="desc 3", test_samples=test_samples[3:6]),
    ]
    test_case_edits = [
        (test_cases[0].name, test_samples[:3]),
        (test_cases[1].name, test_samples[3:6]),
        (test_cases[2].name, test_samples[6:9]),
    ]

    loaded_test_cases = TestCase.init_many(test_case_edits, reset=True)

    assert_test_case(loaded_test_cases[0], f"{name} 1", 1, "desc 1")
    assert_test_case(loaded_test_cases[1], f"{name} 2", 2, "desc 2")
    assert_test_case(loaded_test_cases[2], f"{name} 3", 2, "desc 3")

    verify_test_samples(loaded_test_cases[0], test_samples[:3])
    verify_test_samples(loaded_test_cases[1], test_samples[3:6])
    verify_test_samples(loaded_test_cases[2], test_samples[6:9])
