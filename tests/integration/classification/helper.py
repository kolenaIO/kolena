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

from kolena.classification import TestImage


def assert_test_image_equal(a: TestImage, b: TestImage) -> None:
    assert a.locator == b.locator
    assert a.dataset == b.dataset
    assert a.metadata == b.metadata
    assert sorted(a.labels) == sorted(b.labels)


def assert_test_images_equal(actual: List[TestImage], expected: List[TestImage]) -> None:
    assert len(actual) == len(expected)
    actual = sorted(actual, key=lambda x: x.locator)
    expected = sorted(expected, key=lambda x: x.locator)
    for a, b in zip(actual, expected):
        assert_test_image_equal(a, b)
