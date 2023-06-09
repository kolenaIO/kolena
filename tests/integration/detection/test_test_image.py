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
import uuid
from typing import cast
from typing import List

import pytest

import kolena.detection.ground_truth as ground_truth
from kolena.detection import iter_images
from kolena.detection import load_images
from kolena.detection import TestCase
from kolena.detection import TestImage
from kolena.detection.metadata import Asset
from kolena.detection.metadata import BoundingBox
from kolena.detection.metadata import Landmarks
from tests.integration.detection.helper import assert_test_images_equal
from tests.integration.detection.helper import fake_gt_bounding_box
from tests.integration.detection.helper import fake_gt_classification_label
from tests.integration.detection.helper import fake_gt_segmentation_mask
from tests.integration.detection.helper import fake_label
from tests.integration.detection.helper import fake_points
from tests.integration.helper import fake_random_locator
from tests.integration.helper import with_test_prefix


@pytest.fixture(scope="module")
def test_images() -> List[TestImage]:
    images = fake_test_images(15)
    # images dataset is generated by with_test_prefix
    TestCase(f"{images[0].dataset} test case for registration purposes", images=images)
    return images


def test__load_images__all(test_images: List[TestImage]) -> None:
    result = load_images()
    assert len(result) >= len(test_images)

    result_locators = {image.locator for image in result}
    test_image_locators = {image.locator for image in test_images}
    # regardless of whatever other images have been registered, at least the expected test_images are present
    assert len(test_image_locators - result_locators) == 0


def test__load_images__with_dataset(test_images: List[TestImage]) -> None:
    result = cast(List[TestImage], load_images(test_images[0].dataset))
    assert_test_images_equal(result, test_images)


def test__iter_images__with_dataset(test_images: List[TestImage]) -> None:
    result = list(iter_images(test_images[0].dataset))
    assert_test_images_equal(result, test_images)


def test__resolve_existing() -> None:
    def register(images: List[TestImage]) -> None:
        TestCase(str(uuid.uuid4()), images=images)

    dataset = with_test_prefix(str(uuid.uuid4()))
    image_a0 = TestImage(
        fake_random_locator(),
        dataset=dataset,
        ground_truths=[fake_gt_classification_label(), fake_gt_segmentation_mask()],
    )
    image_b0 = TestImage(
        fake_random_locator(),
        dataset=dataset,
        ground_truths=[fake_gt_bounding_box(), fake_gt_bounding_box()],
    )
    image_c0 = TestImage(fake_random_locator(), dataset=dataset, ground_truths=[])
    register([image_a0, image_b0, image_c0])

    image_a1 = TestImage(image_a0.locator, dataset=dataset)
    image_b1 = TestImage(
        image_b0.locator,
        dataset=dataset,
        ground_truths=[image_b0.ground_truths[1], fake_gt_bounding_box()],
    )
    image_c1 = TestImage(image_c0.locator, dataset=dataset, ground_truths=[fake_gt_segmentation_mask()])
    register([image_a1, image_b1, image_c1])

    assert_test_images_equal(
        load_images(dataset),
        [
            image_a0,
            TestImage(
                image_b0.locator,
                dataset=dataset,
                ground_truths=[image_b0.ground_truths[0], *image_b1.ground_truths],
            ),
            image_c1,
        ],
    )


def test__load_images__metadata() -> None:
    metadata = dict(
        example_str="some example string with\narbitrary\tcharacters 😁",
        example_float=1.2,  # relatively round; no guarantee of exactness
        example_int=-3,
        example_bool=True,
        example_bounding_box=BoundingBox((1, 2), (3, 4)),
        example_landmarks=Landmarks([(1, 2), (3, 4), (5, 6), (7, 8), (9, 0)]),
        example_asset=Asset("s3://path/to/example/asset.jpg"),
    )
    dataset = with_test_prefix(str(uuid.uuid4()))
    image = TestImage(fake_random_locator(), dataset=dataset, ground_truths=[fake_gt_bounding_box()], metadata=metadata)
    TestCase(with_test_prefix(str(uuid.uuid4())), images=[image])
    assert load_images(dataset) == [image]


def test__load_difficult_ground_truth() -> None:
    difficult_classification = ground_truth.ClassificationLabel(fake_label(), difficult=True)
    non_difficult_classification = ground_truth.ClassificationLabel(fake_label(), difficult=False)
    difficult_bbox = ground_truth.BoundingBox(fake_label(), *fake_points(2), difficult=True)
    non_difficult_bbox = ground_truth.BoundingBox(fake_label(), *fake_points(2), difficult=False)
    difficult_seg_mask = ground_truth.SegmentationMask(fake_label(), fake_points(4), difficult=True)
    non_difficult_seg_mask = ground_truth.SegmentationMask(fake_label(), fake_points(4), difficult=False)

    for gts in (
        [difficult_classification, non_difficult_classification],
        [difficult_bbox, non_difficult_bbox],
        [difficult_seg_mask, non_difficult_seg_mask],
    ):
        dataset = with_test_prefix(str(uuid.uuid4()))
        image = TestImage(fake_random_locator(), dataset=dataset, ground_truths=gts)
        TestCase(with_test_prefix(str(uuid.uuid4())), images=[image])
        # note: this method is deprecated
        loaded_images = load_images(dataset)
        assert_test_images_equal(cast(List[TestImage], loaded_images), [image])


def test__load_duplicated_ground_truth() -> None:
    locator = fake_random_locator()
    label = fake_label()
    points = fake_points(2)

    difficult_bbox = ground_truth.BoundingBox(label=label, top_left=points[0], bottom_right=points[1], difficult=True)
    non_difficult_bbox = ground_truth.BoundingBox(
        label=label,
        top_left=points[0],
        bottom_right=points[1],
        difficult=False,
    )

    # Register single ground truth, get single ground truth
    test_case = TestCase(
        with_test_prefix(str(uuid.uuid4())),
        images=[
            TestImage(
                locator,
                ground_truths=[
                    non_difficult_bbox,
                ],
            ),
        ],
    )
    got_images = test_case.load_images()
    assert len(got_images) == 1
    assert len(got_images[0].ground_truths) == 1
    assert got_images[0].ground_truths[0] == non_difficult_bbox

    # Register single ground truth with difficult, get single ground truth
    test_case = TestCase(
        with_test_prefix(str(uuid.uuid4())),
        images=[
            TestImage(
                locator,
                ground_truths=[
                    difficult_bbox,
                ],
            ),
        ],
    )
    got_images = test_case.load_images()
    assert len(got_images) == 1
    assert len(got_images[0].ground_truths) == 1
    assert got_images[0].ground_truths[0] == difficult_bbox

    # Check that nothing changed with non-difficult case
    test_case = TestCase(
        with_test_prefix(str(uuid.uuid4())),
        images=[
            TestImage(
                locator,
                ground_truths=[
                    non_difficult_bbox,
                ],
            ),
        ],
    )
    got_images = test_case.load_images()
    assert len(got_images) == 1
    assert len(got_images[0].ground_truths) == 1
    assert got_images[0].ground_truths[0] == non_difficult_bbox


def fake_test_images(n: int) -> List[TestImage]:
    dataset = with_test_prefix(f"dataset-{str(uuid.uuid4())}")
    gt_choices = [fake_gt_classification_label, fake_gt_bounding_box, fake_gt_segmentation_mask]
    return [
        TestImage(
            fake_random_locator(dataset),
            dataset=dataset,
            ground_truths=[random.choice(gt_choices)() for _ in range(random.randint(0, 3))],
        )
        for _ in range(n)
    ]
