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
import random
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import patch

import pytest

import kolena
from kolena._api.v1.detection import CustomMetrics
from kolena._api.v1.detection import TestRun as TestRunAPI
from kolena.detection import InferenceModel
from kolena.detection import Model
from kolena.detection import test
from kolena.detection import TestCase
from kolena.detection import TestImage
from kolena.detection import TestSuite
from kolena.detection.ground_truth import BoundingBox as GTBoundingBox
from kolena.detection.inference import BoundingBox
from kolena.detection.inference import ClassificationLabel
from kolena.detection.inference import Inference
from kolena.detection.inference import SegmentationMask
from kolena.detection.test_run import TestRun
from kolena.errors import CustomMetricsException
from kolena.errors import InputValidationError
from kolena.errors import RemoteError
from tests.integration.detection.conftest import TestData
from tests.integration.helper import fake_random_locator
from tests.integration.helper import with_test_prefix


@pytest.fixture(scope="module")
def detection_model(detection_test_data: TestData) -> Model:
    return detection_test_data.models[0]


@pytest.fixture(scope="module")
def detection_test_suites(detection_test_data: TestData) -> List[TestSuite]:
    return [detection_test_data.test_suites[0], detection_test_data.test_suites[2]]


@pytest.fixture
def detection_test_image_locators(detection_test_data: TestData) -> List[str]:
    # all image locators in test suite A
    return sorted(detection_test_data.locators[:-1])


def generate_image_results(images: List[TestImage]) -> List[Tuple[TestImage, Optional[List[Inference]]]]:
    return [(image, generate_single_image_inferences(image)) for image in images]


def generate_single_image_inferences(image: TestImage) -> List[Inference]:
    # deterministically generate inferences
    random.seed(hash(image.locator))
    return_switch = random.random()
    if return_switch < 0.2:
        return []
    class_inf = ClassificationLabel(label="car", confidence=random.random())
    bb_inf = BoundingBox(
        confidence=random.random(),
        label="bike",
        top_left=(random.random() * 300, random.random() * 300),
        bottom_right=(random.random() * 300, random.random() * 300),
    )
    seg_inf = SegmentationMask(
        confidence=random.random(),
        label="pedestrian",
        points=[(random.random() * 300, random.random() * 300) for _ in range(5)],
    )
    if return_switch < 0.4:
        return [class_inf]
    if return_switch < 0.6:
        return [bb_inf]
    if return_switch < 0.8:
        return [seg_inf]
    return [class_inf, bb_inf, seg_inf]


#
# Interacting with a TestRun is naturally a sequenced operation -- here each test depends on the next test and likely
# uses some of the same functionality
#


def test__create_or_retrieve(detection_test_data: TestData) -> None:
    model = detection_test_data.models[0]
    test_suite = detection_test_data.test_suites[0]

    with TestRun(model, test_suite) as test_run_created:
        created_id = test_run_created._id

    with TestRun(model, test_suite) as test_run_retrieved:
        retrieved_id = test_run_retrieved._id

    assert retrieved_id == created_id


def test__create_or_retrieve__with_params(detection_test_data: TestData) -> None:
    model = Model(with_test_prefix(f"{__file__}::test_create_or_retrieve_test_run_with_params model"))
    test_suite = detection_test_data.test_suites[0]

    # Check invalid kwargs
    with pytest.raises(InputValidationError):
        TestRun(model, test_suite, test_config=kolena.detection.test_config.FixedGlobalThreshold(-0.5))
    with pytest.raises(InputValidationError):
        TestRun(model, test_suite, test_config=kolena.detection.test_config.FixedGlobalThreshold(1.5))
    with pytest.raises(InputValidationError):
        TestRun(model, test_suite, test_config=kolena.detection.test_config.F1Optimal(-0.5))
    with pytest.raises(InputValidationError):
        TestRun(model, test_suite, test_config=kolena.detection.test_config.F1Optimal(1.5))

    with TestRun(
        model,
        test_suite,
        test_config=kolena.detection.test_config.FixedGlobalThreshold(0.5),
    ) as test_run_created:
        created_id = test_run_created._id

    with TestRun(
        model,
        test_suite,
        test_config=kolena.detection.test_config.FixedGlobalThreshold(0.5),
    ) as test_run_retrieved:
        retrieved_id = test_run_retrieved._id

    assert retrieved_id == created_id


@pytest.mark.depends(on=["test__create_or_retrieve"])
def test__load_images(
    detection_model: Model,
    detection_test_suites: List[TestSuite],
    detection_test_image_locators: List[str],
) -> None:
    with TestRun(detection_model, detection_test_suites[0]) as test_run:
        remaining_images_actual = test_run.load_images()
        assert sorted(image.locator for image in remaining_images_actual) == detection_test_image_locators
        assert sorted(image.metadata["i"] for image in remaining_images_actual) == list(range(4))

    # fetching again should retrieve the same data if no results were uploaded
    with TestRun(detection_model, detection_test_suites[0]) as test_run:
        remaining_images_actual = test_run.load_images(batch_size=500)
        assert sorted(image.locator for image in remaining_images_actual) == detection_test_image_locators

    batch_size = 2
    with TestRun(detection_model, detection_test_suites[0]) as test_run:
        remaining_images_actual = test_run.load_images(batch_size=2)
        assert len(remaining_images_actual) == batch_size

    # zero-size batches are not allowed
    with pytest.raises(InputValidationError):
        with TestRun(detection_model, detection_test_suites[0]) as test_run:
            test_run.load_images(batch_size=0)


@pytest.mark.depends(on=["test__create_or_retrieve"])
def test__iter_images(
    detection_model: Model,
    detection_test_suites: List[TestSuite],
    detection_test_image_locators: List[str],
) -> None:
    with TestRun(detection_model, detection_test_suites[0]) as test_run:
        remaining_images_actual = list(test_run.iter_images())
        assert sorted(image.locator for image in remaining_images_actual) == detection_test_image_locators

    # fetching again should retrieve the same data if no results were uploaded
    with TestRun(detection_model, detection_test_suites[0]) as test_run:
        remaining_images_actual = list(test_run.iter_images())
        assert sorted(image.locator for image in remaining_images_actual) == detection_test_image_locators


@pytest.mark.depends(on=["test__load_images"])
def test__add_inferences__validation(detection_model: Model, detection_test_suites: List[TestSuite]) -> None:
    fake_image = TestImage(fake_random_locator())
    fake_inference = ClassificationLabel(label="car", confidence=0.5)

    with pytest.raises(InputValidationError):
        # assert that we guard against images from outside the test suite
        with TestRun(detection_model, detection_test_suites[0]) as test_run:
            test_run.add_inferences(fake_image, [fake_inference])


@pytest.mark.depends(on=["test__load_images"])
def test__add_inferences__validation__invalid_confidence(
    detection_model: Model,
    detection_test_suites: List[TestSuite],
) -> None:
    with pytest.raises(RemoteError):
        with TestRun(detection_model, detection_test_suites[0]) as test_run:
            [image] = test_run.load_images(batch_size=1)
            bad_inference = ClassificationLabel("car", 0)
            bad_inference.confidence = float("nan")  # bypass validation on constructor
            test_run.add_inferences(image, [bad_inference])


@pytest.mark.depends(on=["test__load_images"])
def test_add_inferences__validation__ignored_sample() -> None:
    test_name = with_test_prefix(f"{__file__}::test_add_inferences__validation__ignored_sample")
    model = Model(f"{test_name} model")
    images = [
        TestImage(
            fake_random_locator(),
            ground_truths=[
                kolena.detection.ground_truth.BoundingBox(label="car", top_left=(0, 0), bottom_right=(100, 100)),
            ],
        )
        for _ in range(5)
    ]
    test_case = TestCase(f"{test_name} test_case", images=images)
    test_suite = TestSuite(f"{test_name} test_suite", test_cases=[test_case])

    with pytest.raises(RemoteError):
        with TestRun(model, test_suite) as test_run:
            for i, image in enumerate(test_run.iter_images()):
                test_run.add_inferences(
                    image,
                    inferences=[BoundingBox(label="car", confidence=0.5, top_left=(0, 0), bottom_right=(100, 100))],
                )
                if i == 0:
                    test_run.add_inferences(image, inferences=None)


@pytest.mark.depends(on=["test__load_images"])
def test__add_inferences__validation__all_ignore() -> None:
    test_name = with_test_prefix(f"{__file__}::test__add_inferences__validation__all_ignore")
    model = Model(f"{test_name} model")
    images = [
        TestImage(
            fake_random_locator(),
            ground_truths=[
                kolena.detection.ground_truth.BoundingBox(label="car", top_left=(0, 0), bottom_right=(100, 100)),
            ],
        )
        for _ in range(5)
    ]
    test_case = TestCase(f"{test_name} test_case", images=images)
    test_suite = TestSuite(f"{test_name} test_suite", test_cases=[test_case])

    with pytest.raises(RemoteError):
        with TestRun(model, test_suite) as test_run:
            for image in test_run.iter_images():
                test_run.add_inferences(image, inferences=None)


@pytest.mark.depends(on=["test__add_inferences__validation"])
def test__add_inferences(detection_model: Model, detection_test_suites: List[TestSuite]) -> None:
    with TestRun(detection_model, detection_test_suites[0]) as test_run:
        remaining_images = test_run.load_images(batch_size=2)
        image_results = generate_image_results(remaining_images)
        for image, inferences in image_results:
            test_run.add_inferences(image, inferences)

    with pytest.raises(InputValidationError):
        # shouldn't be able to upload duplicate entries
        with TestRun(detection_model, detection_test_suites[0]) as test_run:
            for image, inferences in image_results:
                test_run.add_inferences(image, inferences)

    # test no inferences and iter_images
    with TestRun(detection_model, detection_test_suites[0]) as test_run:
        for image in test_run.iter_images():
            test_run.add_inferences(image, [])
            break  # break to process only one image

    with TestRun(detection_model, detection_test_suites[0]) as test_run:
        remaining_images = test_run.load_images()
        image_results = generate_image_results(remaining_images)
        # load images that have not been processed such that results can be uploaded
        for _image, _inferences in image_results:
            test_run.add_inferences(_image, _inferences)

    # results have been received for the entire suite, no more images to fetch
    with TestRun(detection_model, detection_test_suites[0]) as test_run:
        remaining_images = test_run.load_images()
        assert len(remaining_images) == 0


@pytest.mark.depends(on=["test__add_inferences"])
def test__noop(detection_test_data: TestData) -> None:
    model = detection_test_data.models[0]
    # test-suite "A_subset"
    with TestRun(model, detection_test_data.test_suites[3]) as test_run:
        remaining_images = test_run.load_images()
        assert len(remaining_images) == 0


def test__test(detection_test_data: TestData) -> None:
    model = InferenceModel(with_test_prefix(f"{__file__}::test__test inference model"), infer=lambda _image: [])
    test_suite = detection_test_data.test_suites[3]

    with TestRun(model, test_suite) as test_run:
        test_run_id = test_run._id
        assert len(test_run.load_images()) > 0

    # should complete all tests
    test(model, test_suite)

    with TestRun(model, test_suite) as test_run:
        assert test_run_id == test_run._id
        assert len(test_run.load_images()) == 0


def test__test__reset() -> None:
    test_name = with_test_prefix(f"{__file__}::test__test__reset")
    n_images = 5
    images = [
        TestImage(
            fake_random_locator(),
            dataset=test_name,
            ground_truths=[
                GTBoundingBox(
                    "bike",
                    (0.0 + idx, 0.0 + idx),
                    (100.0 + idx, 100.0 + idx),
                ),
                GTBoundingBox(
                    "pedestrian",
                    (0.0 + idx * 10, 0.0 + idx * 10),
                    (100.0 + idx * 10, 100.0 + idx * 10),
                ),
            ],
        )
        for idx in range(n_images)
    ]
    test_case = TestCase(f"{test_name} test_case", images=images)
    test_suite = TestSuite(name=f"{test_name} test suite", test_cases=[test_case])

    bb_bike = BoundingBox(
        label="bike",
        confidence=0.89,
        top_left=(42.0, 42.0),
        bottom_right=(420.0, 420.0),
    )

    def infer_bike(_: TestImage) -> List[BoundingBox]:
        return [bb_bike]

    bb_pedestrian = BoundingBox(
        label="pedestrian",
        confidence=0.79,
        top_left=(42.0, 42.0),
        bottom_right=(420.0, 420.0),
    )

    def infer_pedestrian(_: TestImage) -> List[BoundingBox]:
        return [bb_pedestrian]

    model_bike = InferenceModel(f"{test_name} inference model", infer=infer_bike)
    model_pedestrian = InferenceModel(f"{test_name} inference model", infer=infer_pedestrian)
    assert model_bike._id == model_pedestrian._id

    with TestRun(model_bike, test_suite) as test_run:
        assert len(test_run.load_images()) > 0

    test(model_bike, test_suite)
    assert [inf for _, inf in model_bike.load_inferences(test_suite)] == [[bb_bike] for _ in range(n_images)]

    with TestRun(model_bike, test_suite) as test_run:
        assert len(test_run.load_images()) == 0

    test(model_pedestrian, test_suite, reset=True)
    assert [inf for _, inf in model_pedestrian.load_inferences(test_suite)] == [
        [bb_pedestrian] for _ in range(n_images)
    ]


def test__custom_metrics(detection_test_data: TestData) -> None:
    def custom_metrics(inferences: List[Tuple[TestImage, Optional[List[Inference]]]]) -> CustomMetrics:
        num_infers = sum(len(infer) if infer else 0 for sample, infer in inferences)
        return {"foo": num_infers}

    model = InferenceModel(
        with_test_prefix(f"{__file__}::test_test_run_custom_metrics"),
        infer=lambda _image: [
            BoundingBox(
                confidence=random.random(),
                label="car",
                top_left=(random.random() * 300, random.random() * 300),
                bottom_right=(random.random() * 300, random.random() * 300),
            ),
        ],
    )
    test_suite = detection_test_data.test_suites[1]
    test(model, test_suite, custom_metrics_callback=custom_metrics)


def test__custom_metrics__error(detection_test_data: TestData) -> None:
    def bad_custom_metrics(_: List[Tuple[TestImage, Optional[List[Inference]]]]) -> CustomMetrics:
        raise KeyError("dumb error")

    model = InferenceModel(
        with_test_prefix(f"{__file__}::test_test_run_custom_metrics_error"),
        infer=lambda _image: [
            BoundingBox(
                confidence=random.random(),
                label="car",
                top_left=(random.random() * 300, random.random() * 300),
                bottom_right=(random.random() * 300, random.random() * 300),
            ),
        ],
    )
    test_suite = detection_test_data.test_suites[0]

    with pytest.raises(CustomMetricsException):
        test(model, test_suite, custom_metrics_callback=bad_custom_metrics)


def test__mark_crashed(detection_test_data: TestData) -> None:
    def infer(_: TestImage) -> Optional[List[BoundingBox]]:
        raise RuntimeError

    model = InferenceModel(with_test_prefix(f"{__file__}::test_mark_crashed inference model"), infer=infer)
    test_suite = detection_test_data.test_suites[1]

    test_run = TestRun(model, test_suite)

    with patch("kolena.detection._internal.test_run.report_crash") as patched:
        with pytest.raises(RuntimeError):
            with test_run:
                raise RuntimeError

    patched.assert_called_once_with(test_run._id, TestRunAPI.Path.MARK_CRASHED)

    with patch("kolena.detection._internal.test_run.report_crash") as patched:
        with pytest.raises(RuntimeError):
            test(model, test_suite)

    patched.assert_called_once_with(test_run._id, TestRunAPI.Path.MARK_CRASHED)
