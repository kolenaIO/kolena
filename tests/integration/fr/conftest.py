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
from typing import Tuple

import numpy as np
import pytest as pytest

from kolena.fr import Model
from kolena.fr import TestCase
from kolena.fr import TestImages
from kolena.fr import TestSuite
from kolena.fr.datatypes import TestCaseRecord
from tests.integration.helper import with_test_prefix

DATASET = "test_dummy_dataset"


def get_test_locator(idx: int) -> str:
    return f"s3://sdk-server-integration-tests-fr/{DATASET}/{idx}.jpg"


@pytest.fixture(scope="session")
def test_locators() -> List[str]:
    return [get_test_locator(idx) for idx in range(3)]


@pytest.fixture(scope="session")
def register_test_samples(test_locators: List[str]) -> None:
    with TestImages.register() as registrar:
        for i, locator in enumerate(test_locators):
            registrar.add(locator, DATASET, 250, 250, tags=dict(age=i))


@pytest.fixture(scope="session")
def test_samples(test_locators: List[str]) -> List[TestCaseRecord]:
    test_samples = []
    for locator_a in test_locators:
        for locator_b in test_locators:
            # 0, 4, 8 is the genuine pair
            test_samples.append((locator_a, locator_b, locator_a == locator_b))
    return test_samples


@pytest.fixture(scope="session")
def fr_images() -> Tuple[List[str], List[TestCaseRecord]]:
    locators = [f"s3://fake-bucket/{i}.png" for i in range(6)]
    locator_a = [*locators, locators[0], locators[1], locators[0], locators[1]]
    locator_b = [*locators[1:], locators[0], locators[2], locators[5], locators[0], locators[1]]
    is_same = [True, False, False, False, False, False, False, True, True, True]

    with TestImages.register() as registrar:
        for i, locator in enumerate(locators):
            registrar.add(
                locator,
                "fake-data-source",
                (i + 1) * 100,
                (i + 1) * 100,
                bounding_box=np.array([10, 10, 20, 20]),
                landmarks=np.array([10, 10, 20, 20, 30, 30, 40, 40, 50, 50]),
                tags=None,
            )

    return locators, list(zip(locator_a, locator_b, is_same))


@pytest.fixture(scope="session")
def fr_models() -> List[Model]:
    return [Model.create(with_test_prefix(name), {"some": "metadata", "one": 1, "false": False}) for name in ["a", "b"]]


@pytest.fixture(scope="session")
def fr_test_cases(fr_images: Tuple[List[str], List[TestCaseRecord]]) -> List[TestCase]:
    image_pairs = fr_images[1]
    test_case_name_a = with_test_prefix("A")
    test_case_a = TestCase(
        test_case_name_a,
        description="filter",
        test_samples=[
            image_pairs[0],
            image_pairs[1],
            image_pairs[8],
            image_pairs[9],
        ],
    )
    test_case_a_updated = TestCase(
        test_case_name_a,
        description="description",
        test_samples=[
            image_pairs[0],
            image_pairs[1],
            image_pairs[3],
        ],
        reset=True,
    )

    test_case_name_b = with_test_prefix("B")
    test_case_b = TestCase(test_case_name_b, description="fields", test_samples=[image_pairs[1], image_pairs[2]])
    test_case_b_updated = TestCase(
        test_case_name_b,
        description="etc",
        test_samples=[image_pairs[2], image_pairs[6], image_pairs[7]],
    )

    test_case_b_sub = TestCase(with_test_prefix("B_subset"), description="and more!", test_samples=[image_pairs[2]])

    return [
        test_case_a,
        test_case_a_updated,
        test_case_b,
        test_case_b_updated,
        test_case_b_sub,
    ]


@pytest.fixture(scope="session")
def fr_test_suites(fr_test_cases: List[TestCase]) -> List[TestSuite]:
    test_suite_name_a = with_test_prefix("A")
    test_suite_a = TestSuite(
        test_suite_name_a,
        description="filler",
        baseline_test_cases=[fr_test_cases[0]],
        non_baseline_test_cases=[fr_test_cases[2]],
    )
    test_suite_a_updated = TestSuite(
        test_suite_name_a,
        description="description",
        baseline_test_cases=[fr_test_cases[1]],
        non_baseline_test_cases=[fr_test_cases[2]],
        reset=True,
    )

    test_suite_b = TestSuite(with_test_prefix("B"), description="fields", baseline_test_cases=[fr_test_cases[3]])
    test_suite_a_sub = TestSuite(
        with_test_prefix("A_subset"),
        description="etc",
        baseline_test_cases=[fr_test_cases[4]],
    )

    return [test_suite_a, test_suite_a_updated, test_suite_b, test_suite_a_sub]
