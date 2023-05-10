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
from dataclasses import dataclass
from typing import List

import pytest

from kolena.detection import Model
from kolena.detection import TestCase
from kolena.detection import TestImage
from kolena.detection import TestSuite
from kolena.detection.ground_truth import BoundingBox
from kolena.detection.ground_truth import ClassificationLabel
from kolena.detection.ground_truth import SegmentationMask
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


@dataclass(frozen=True)
class TestData:
    test_cases: List[TestCase]
    test_suites: List[TestSuite]
    models: List[Model]
    locators: List[str]


@pytest.fixture(scope="session")
def detection_test_data() -> TestData:
    ground_truths = [
        ClassificationLabel("car"),
        ClassificationLabel("bike"),
        BoundingBox("boat", top_left=(0.0, 1.5), bottom_right=(0.3, 3.4)),
        SegmentationMask("van", [(4.0, 1.5), (0.9, 3.4), (19.5, 17.6), (8, 8)]),
        BoundingBox("boat", top_left=(50, 60), bottom_right=(60, 100)),
        BoundingBox("pedestrian", top_left=(120, 70), bottom_right=(190, 100)),
        SegmentationMask("truck", [(0, 15), (0.9, 3.4), (19.5, 17.6), (0, 15)]),
        SegmentationMask("airplane", [(4.0, 1.5), (0.9, 3.4), (19.5, 17.6), (8, 8)]),
    ]
    dataset = with_test_prefix("fake-data-set")
    images = [(fake_locator(i, "detection/base"), {"example": "metadata", "i": i}) for i in range(5)]

    test_case_a = TestCase(
        with_test_prefix("A"),
        description="filler",
        images=[
            TestImage(locator=images[0][0], dataset=dataset, metadata=images[0][1], ground_truths=[ground_truths[0]]),
            TestImage(locator=images[1][0], dataset=dataset, metadata=images[1][1]),
        ],
    )
    test_case_a_updated = TestCase(
        with_test_prefix("A"),
        description="description",
        images=[
            TestImage(locator=images[0][0], dataset=dataset, metadata=images[0][1], ground_truths=[ground_truths[0]]),
            TestImage(locator=images[1][0], dataset=dataset, metadata=images[1][1]),
            TestImage(locator=images[2][0], dataset=dataset, metadata=images[2][1], ground_truths=[ground_truths[2]]),
            TestImage(locator=images[4][0], dataset=dataset, metadata=images[4][1]),
        ],
        reset=True,
    )
    test_case_b = TestCase(
        with_test_prefix("B"),
        description="fields",
        images=[
            TestImage(
                locator=images[2][0],
                dataset=dataset,
                metadata=images[2][1],
                ground_truths=[ground_truths[1], ground_truths[2]],
            ),
            TestImage(locator=images[3][0], dataset=dataset, metadata=images[3][1], ground_truths=[ground_truths[4]]),
        ],
    )
    test_case_b_updated = TestCase(
        with_test_prefix("B"),
        description="etc",
        images=[
            TestImage(locator=images[1][0], dataset=dataset, metadata=images[1][1]),
            TestImage(
                locator=images[2][0],
                dataset=dataset,
                metadata=images[2][1],
                ground_truths=[
                    ground_truths[2],
                    ground_truths[3],
                ],
            ),
            TestImage(
                locator=images[3][0],
                dataset=dataset,
                metadata=images[3][1],
                ground_truths=[
                    ground_truths[5],
                    ground_truths[7],
                ],
            ),
        ],
        reset=True,
    )
    test_case_b_subset = TestCase(
        with_test_prefix("B_subset"),
        description="and more!",
        images=[
            TestImage(locator=images[3][0], dataset=dataset, metadata=images[3][1], ground_truths=[ground_truths[6]]),
        ],
    )

    test_cases = [test_case_a, test_case_a_updated, test_case_b, test_case_b_updated, test_case_b_subset]

    test_suite_name_a = with_test_prefix("A")
    test_suite_a = TestSuite(test_suite_name_a, description="filler", test_cases=[test_case_a, test_case_b])
    test_suite_a_updated = TestSuite(
        test_suite_name_a,
        description="description",
        test_cases=[test_case_a_updated, test_case_b],
        reset=True,
    )
    test_suite_b = TestSuite(with_test_prefix("B"), description="fields", test_cases=[test_case_b_updated])
    test_suite_a_subset = TestSuite(
        with_test_prefix("A_subset"),
        description="etc",
        test_cases=[test_case_b_subset],
    )

    test_suites = [test_suite_a, test_suite_a_updated, test_suite_b, test_suite_a_subset]

    models = [
        Model(with_test_prefix("a"), metadata={"some": "metadata"}),
        Model(with_test_prefix("b"), metadata={"one": 1, "false": False}),
    ]

    return TestData(test_cases=test_cases, test_suites=test_suites, models=models, locators=[img[0] for img in images])


pytest.register_assert_rewrite("tests.integration.detection.helper")
