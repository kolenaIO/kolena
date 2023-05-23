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
from pydantic import ValidationError

from kolena._api.v1.workflow import WorkflowType
from kolena.detection import Model
from kolena.detection import TestCase
from kolena.detection import TestImage
from kolena.detection import TestRun
from kolena.detection import TestSuite
from kolena.detection.ground_truth import BoundingBox as GTBoundingBox
from kolena.detection.ground_truth import GroundTruth
from kolena.detection.ground_truth import SegmentationMask as GTSegmentationMask
from kolena.detection.inference import BoundingBox as InfBoundingBox
from kolena.detection.inference import Inference
from kolena.detection.inference import SegmentationMask as InfSegmentationMask
from tests.integration.detection.conftest import TestData
from tests.integration.detection.helper import assert_test_images_equal
from tests.integration.detection.helper import fake_confidence
from tests.integration.helper import fake_random_locator
from tests.integration.helper import with_test_prefix


def test__init() -> None:
    name = with_test_prefix(f"{__file__}::test_init model")
    metadata = dict(a="A", b=1, c=None, d=True, e=[1, "test", None], f=dict(g="test"), h=None)
    model = Model(name, metadata=metadata)
    assert model.name == name
    assert model.metadata == metadata
    assert model._workflow == WorkflowType.DETECTION
    assert model == Model(name)


def test__init__changed_metadata() -> None:
    name = with_test_prefix(f"{__file__}::test_init_changed_metadata model")
    metadata = dict(one="two", three="four")
    model = Model(name, metadata=metadata)
    assert model == Model(name, metadata=dict(changed="metadata"))  # changed metadata is ignored


def test__create__bad_metadata() -> None:
    with pytest.raises(ValidationError):
        Model(with_test_prefix(f"{__file__}::test_create_bad_metadata model"), "not a dict")  # type: ignore
    with pytest.raises(ValidationError):
        Model(
            with_test_prefix(f"{__file__}::test_create_bad_metadata model 2"),
            ["also", "not", "a", "dict"],
        )  # type: ignore


def test__load_inferences__no_inferences(detection_test_data: TestData) -> None:
    model = detection_test_data.models[0]
    test_suite = detection_test_data.test_suites[1]
    test_cases = test_suite.test_cases
    test_case_id_0 = test_suite.test_cases[0]._id
    test_case_id_1 = test_suite.test_cases[1]._id

    inferences = model.load_inferences(test_suite)
    assert len(inferences) == 5

    inferences_0 = model.load_inferences(test_cases[0])
    assert len(inferences_0) == 4

    inferences_1 = model.load_inferences(test_cases[1])
    assert len(inferences_1) == 2

    inferences = model.load_inferences_by_test_case(test_suite)
    assert [(test_case_id, len(infer)) for test_case_id, infer in inferences.items()] == [
        (test_case_id_0, 4),
        (test_case_id_1, 2),
    ]
    # verify ground_truths are properly scoped per test-case
    # test-cases[0] has sample (sample_0, sample_1, sample_2, sample_4), gt (gt_0, None, gt_2, None,)
    # test-cases[1] has sample (sample_2, sample_2, sample_3), gt (gt_1, gt_2, gt_4,)
    # test sample #3 across test suite, should have different ground_truths
    sample, _ = sorted(inferences[test_case_id_0], key=lambda x: x[0].locator)[2]
    assert len(sample.ground_truths) == 1
    sample, _ = sorted(inferences[test_case_id_1], key=lambda x: x[0].locator)[0]
    assert len(sample.ground_truths) == 2

    # extra check for behavior consistency
    assert inferences[test_case_id_0] == inferences_0
    assert inferences[test_case_id_1] == inferences_1


def _test_load_inferences(test_name: str, n_images: int, gts: List[GroundTruth], infs: List[Inference]) -> None:
    model = Model(with_test_prefix(f"{test_name} model"))
    images = [
        TestImage(
            fake_random_locator("detection/test-model"),
            ground_truths=gts,
        )
        for _ in range(n_images)
    ]
    test_case = TestCase(with_test_prefix(f"{__file__}::{test_name} test_case"), images=images)
    test_suite = TestSuite(with_test_prefix(f"{__file__}::{test_name} test_suite"), test_cases=[test_case])
    fake_inferences = []
    with TestRun(model, test_suite) as test_run:
        for image in test_run.iter_images():
            fake_inferences.append(infs)
            test_run.add_inferences(image, inferences=infs)

    # fetch inference to make sure it has all inferences
    inferences = model.load_inferences(test_suite)
    actual_images = [test_image for test_image, _ in inferences]
    actual_inferences = [infer for _, infer in inferences]
    assert_test_images_equal(images, actual_images)
    assert fake_inferences == actual_inferences

    # fetch inference using load_inferences_by_test_case to make sure it has all inferences
    inferences_by_test_case = model.load_inferences_by_test_case(test_suite)
    assert len(inferences_by_test_case) == 1

    actual_images = [test_image for _, infer in inferences_by_test_case.items() for test_image, _ in infer]
    actual_inferences = [inf for _, infer in inferences_by_test_case.items() for _, inf in infer]
    assert_test_images_equal(images, actual_images)
    assert fake_inferences == actual_inferences


def test__load_inferences__bounding_box() -> None:
    _test_load_inferences(
        test_name="test__load_inferences__bounding_box",
        n_images=5,
        gts=[
            GTBoundingBox(
                label="car",
                top_left=(0, 0),
                bottom_right=(50, 50),
            ),
            GTBoundingBox(
                label="bus",
                top_left=(20, 20),
                bottom_right=(100, 100),
            ),
        ],
        infs=[
            InfBoundingBox(
                label="car",
                confidence=fake_confidence(),
                top_left=(0, 0),
                bottom_right=(30, 50),
            ),
            InfBoundingBox(
                label="bus",
                confidence=fake_confidence(),
                top_left=(20, 10),
                bottom_right=(80, 80),
            ),
        ],
    )


def test__load_inferences__segmentation_mask() -> None:
    _test_load_inferences(
        test_name="test__load_inferences__segmentation_mask",
        n_images=5,
        gts=[
            GTSegmentationMask(
                label="car",
                points=[(0, 0), (1, 1), (2, 2)],
            ),
            GTSegmentationMask(
                label="bus",
                points=[(0, 0), (1, 1), (2, 2)],
            ),
        ],
        infs=[
            InfSegmentationMask(
                label="car",
                confidence=fake_confidence(),
                points=[(0, 0), (1, 1), (2, 2)],
            ),
            InfSegmentationMask(
                label="bus",
                confidence=fake_confidence(),
                points=[(0, 0), (1, 1), (2, 2)],
            ),
        ],
    )
