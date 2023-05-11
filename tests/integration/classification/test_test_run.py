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

import pytest
from integration.helper import fake_locator
from integration.helper import with_test_prefix

from kolena.classification import InferenceModel
from kolena.classification import test
from kolena.classification import TestCase
from kolena.classification import TestImage
from kolena.classification import TestRun
from kolena.classification import TestSuite
from kolena.detection import InferenceModel as DetectionModel

N_IMAGES = 5


def assert_locators_equal(images1: List[TestImage], images2: List[TestImage]) -> None:
    def locators(imgs: List[TestImage]) -> List[str]:
        return sorted([img.locator for img in imgs])

    assert locators(images1) == locators(images2)


def create_test_image(index: int, label: str = "dog") -> TestImage:
    dataset = with_test_prefix(f"{__file__}::test_images fixture dataset")
    return TestImage(
        locator=fake_locator(index, "classification/test-run"),
        dataset=dataset,
        labels=[label],
        metadata={},
    )


@pytest.fixture(scope="module")
def test_images() -> List[TestImage]:
    return [create_test_image(idx) for idx in range(N_IMAGES)]


@pytest.fixture(scope="module")
def test_case(test_images: List[TestImage]) -> TestCase:
    name = with_test_prefix(f"{__file__}::test_case fixture test case")
    return TestCase(name, description="test case description", images=test_images)


@pytest.fixture(scope="module")
def test_suite(test_case: TestCase) -> TestSuite:
    name = with_test_prefix(f"{__file__}::test_suite fixture test suite")
    return TestSuite(name, test_cases=[test_case])


@pytest.fixture(scope="module")
def test_images_1() -> List[TestImage]:
    offset = N_IMAGES
    return [create_test_image(idx, "cat") for idx in range(offset, offset + N_IMAGES)]


@pytest.fixture(scope="module")
def test_case_1(test_images_1: List[TestImage]) -> TestCase:
    name = with_test_prefix(f"{__file__}::TestCase fixture test case")
    return TestCase(name, description="test case description", images=test_images_1)


@pytest.fixture(scope="module")
def test_suite_1(test_case_1: TestCase) -> TestSuite:
    name = with_test_prefix(f"{__file__}::test_suite_1 fixture test suite")
    return TestSuite(name, test_cases=[test_case_1])


@pytest.fixture(scope="module")
def test_models() -> List[InferenceModel]:
    """same model with different infer"""

    def infer_dog_0(_: TestImage) -> List[Tuple[str, float]]:
        return [("dog", 0.89)]

    def infer_dog_1(_: TestImage) -> List[Tuple[str, float]]:
        return [("dog", 0.79)]

    def infer_cat(_: TestImage) -> List[Tuple[str, float]]:
        return [("cat", 0.69)]

    name = f"{__file__}::test_model fixture test model"
    model_dog_0 = InferenceModel(
        name,
        infer=infer_dog_0,
    )
    model_dog_1 = InferenceModel(
        name,
        infer=infer_dog_1,
    )
    model_cat = InferenceModel(
        name,
        infer=infer_cat,
    )
    return [model_dog_0, model_dog_1, model_cat]


def test__load_images__reset(
    test_suite: TestCase,
    test_models: List[InferenceModel],
    test_images: List[TestImage],
) -> None:
    test_model = test_models[0]

    with TestRun(test_model, test_suite) as test_run:
        remaining_images = test_run.load_images()
        assert_locators_equal(remaining_images, test_images)

    test(test_model, test_suite)

    with TestRun(test_model, test_suite) as test_run:
        remaining_images = test_run.load_images()
        assert remaining_images == []

    with TestRun(test_model, test_suite, reset=True) as test_run:
        remaining_images = test_run.load_images()
        assert_locators_equal(remaining_images, test_images)


def test__test__mismatch_workflow(test_suite: TestCase) -> None:
    name = with_test_prefix(f"{__file__}::test__test__mismatch_workflow model")
    model = DetectionModel(name, infer=lambda _: None)
    with pytest.raises(ValueError) as exc_info:
        test(model, test_suite)

    expected_error_msg = "1 validation error for"
    exc_info_value = str(exc_info.value)
    assert expected_error_msg in exc_info_value


def test__test__reset(
    test_suite: TestCase,
    test_suite_1: TestCase,
    test_models: List[InferenceModel],
) -> None:
    model_dog_0, model_dog_1, model_cat = test_models
    # verify they are the same model with different infer func
    assert model_dog_0._id == model_dog_1._id == model_cat._id

    test(model_cat, test_suite_1)

    test(model_dog_0, test_suite, reset=True)
    target_result = ("dog", 0.89)
    assert [inf for _, inf in model_dog_0.load_inferences(test_suite)] == [[target_result] for _ in range(N_IMAGES)]

    # different labels between inference and ground truth, no results from load_inference
    test(model_cat, test_suite, reset=True)
    target_result = ("cat", 0.69)
    assert [inf for _, inf in model_cat.load_inferences(test_suite)] == [None for _ in range(N_IMAGES)]

    test(model_dog_1, test_suite, reset=True)
    target_result = ("dog", 0.79)
    assert [inf for _, inf in model_dog_1.load_inferences(test_suite)] == [[target_result] for _ in range(N_IMAGES)]
