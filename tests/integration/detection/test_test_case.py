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
from copy import deepcopy
from typing import List

import pytest

import kolena.detection.metadata
from kolena._api.v1.workflow import WorkflowType
from kolena.classification import TestCase as ClassificationTestCase
from kolena.detection import TestCase
from kolena.detection import TestImage
from kolena.detection.ground_truth import BoundingBox
from kolena.detection.ground_truth import ClassificationLabel
from kolena.detection.ground_truth import SegmentationMask
from kolena.detection.metadata import Asset
from kolena.errors import NameConflictError
from kolena.errors import NotFoundError
from kolena.errors import WorkflowMismatchError
from kolena.workflow.annotation import BoundingBox as GenericBoundingBox
from tests.integration.detection.helper import assert_test_images_equal
from tests.integration.generic.dummy import DummyGroundTruth
from tests.integration.generic.dummy import DummyTestSample
from tests.integration.generic.dummy import TestCase as GenericTestCase
from tests.integration.helper import fake_random_locator
from tests.integration.helper import with_test_prefix


@pytest.fixture
def test_dataset() -> List[TestImage]:
    sample_dir = "detection/test-case"
    dataset = with_test_prefix(f"{__file__}::test_dataset fixture dataset")
    return [
        TestImage(fake_random_locator(sample_dir), dataset=dataset),
        TestImage(fake_random_locator(sample_dir), dataset=dataset, ground_truths=[ClassificationLabel("car")]),
        TestImage(fake_random_locator(sample_dir), dataset=dataset, ground_truths=[ClassificationLabel("bike")]),
        TestImage(fake_random_locator(sample_dir), dataset=dataset, ground_truths=[BoundingBox("car", (0, 0), (1, 1))]),
        TestImage(
            fake_random_locator(sample_dir),
            dataset=dataset,
            ground_truths=[
                BoundingBox("car", (0, 0), (1, 1)),
                BoundingBox("car", (2, 2), (3, 4)),
                BoundingBox("bike", (3, 3), (9, 9)),
                BoundingBox("car", (4, 2), (7, 8)),
            ],
        ),
        TestImage(
            fake_random_locator(sample_dir),
            dataset=dataset,
            ground_truths=[
                SegmentationMask(
                    "bike",
                    [
                        (0, 0),
                        (1, 2),
                        (2, 1),
                    ],
                ),
            ],
        ),
    ]


def test__init() -> None:
    name = with_test_prefix(f"{__file__}::test__init test case")
    description = "some\ndescription\n\twith punctuation!"
    test_case = TestCase(name, description=description)
    assert test_case.name == name
    assert test_case.version == 0
    assert test_case.description == description
    assert test_case._workflow == WorkflowType.DETECTION

    test_case2 = TestCase(name)  # should re-load previously created
    assert test_case == test_case2

    test_case3 = TestCase(name, description="a different description")  # different description is ignored
    assert test_case == test_case3


def test__init__with_version(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__with_version test case")
    test_case = TestCase(name, description="test")
    test_case0 = TestCase(name, version=test_case.version)  # reload with matching version
    assert test_case == test_case0

    with pytest.raises(NameConflictError):
        TestCase(name, version=123)  # invalid version throws

    with test_case.edit() as editor:
        editor.add(test_dataset[0])

    assert test_case.version == 1
    assert test_case == TestCase(name, version=test_case.version)
    assert test_case0 == TestCase(name, version=test_case0.version)


def test__init__with_images(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__with_images test case")
    images = [test_dataset[0], test_dataset[3], test_dataset[4]]
    test_case = TestCase(name, images=images)
    assert test_case.version == 1
    assert_test_images_equal(test_case.load_images(), images)
    assert test_case._workflow == WorkflowType.DETECTION


def test__init__reset(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__reset test case")
    description = f"{name} (description)"
    images = [test_dataset[3]]
    TestCase(name, description=description, images=images)

    new_images = [test_dataset[4]]
    test_case = TestCase(name, images=new_images, reset=True)
    assert test_case.version == 2
    assert test_case.description == description  # not updated or cleared
    assert_test_images_equal(test_case.load_images(), new_images)


def test__init__with_images_reset(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__with_images_reset test case")
    images = [test_dataset[0], test_dataset[3], test_dataset[4]]
    test_case = TestCase(name, images=images, reset=True)
    assert test_case.version == 1
    assert_test_images_equal(test_case.load_images(), images)
    assert test_case._workflow == WorkflowType.DETECTION


def test__init__reset_with_overlap(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__reset_with_overlap test case")
    description = f"{name} (description)"
    images_1 = [test_dataset[0], test_dataset[3]]
    images_2 = [test_dataset[0], test_dataset[4]]
    TestCase(name, description=description, images=images_1)

    test_case = TestCase(name, images=images_2, reset=True)
    assert test_case.version == 2
    assert test_case.description == description  # not updated or cleared
    assert_test_images_equal(test_case.load_images(), images_2)  # overlapping should be preserved


def test__init__reset_with_other_test_case(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__reset_with_other_test_case test case")
    name_other = with_test_prefix(f"{__file__}::test__init__reset_with_other_test_case test case (other)")
    description = f"{name} (description)"
    images_1 = [test_dataset[0], test_dataset[3]]
    images_2 = [test_dataset[0], test_dataset[3], test_dataset[4]]
    images_3 = [test_dataset[0]]

    # Create and update test case
    TestCase(name, description=description, images=images_1)
    test_case_other = TestCase(name_other, images=images_2)

    test_case = TestCase(name, images=images_3, reset=True)
    assert test_case.version == 2
    assert test_case.description == description  # not updated or cleared
    assert_test_images_equal(test_case_other.load_images(), images_2)  # images_2 should be untouched
    assert_test_images_equal(test_case.load_images(), images_3)  # images_1 should be cleared


def test__init__reset_resets_all_past_samples(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__init__reset_resets_all_past_samples test case")
    description = f"{name} (description)"
    images_1 = [test_dataset[0], test_dataset[3]]
    images_2 = [test_dataset[0], test_dataset[3], test_dataset[4], test_dataset[5]]
    images_3 = [test_dataset[1], test_dataset[2]]

    # Create and update test case
    initial_test_case = TestCase(name, description=description, images=images_1)
    with initial_test_case.edit() as editor:
        for image in images_2:
            editor.add(image)

    test_case = TestCase(name, images=images_3, reset=True)
    assert test_case.version == 3
    assert test_case.description == description  # not updated or cleared
    assert_test_images_equal(test_case.load_images(), images_3)  # both images_1 and images_2 should be cleared


def test__edit(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit test case")
    test_case = TestCase(name)
    assert test_case.version == 0

    new_description = "updated description"
    with test_case.edit() as editor:
        editor.description(new_description)
        for image in test_dataset:
            editor.add(image)
        editor.remove(test_dataset[-1])

    assert test_case.version == 1
    assert test_case.description == new_description
    images_loaded = test_case.load_images()
    remaining_images = test_dataset[:-1]
    assert_test_images_equal(images_loaded, remaining_images)

    with test_case.edit() as editor:
        editor.remove(remaining_images[0])
        editor.remove(remaining_images[1])

    assert test_case.version == 2
    images_loaded = test_case.load_images()
    remaining_images = remaining_images[2:]
    assert_test_images_equal(images_loaded, remaining_images)


def test__edit__reset(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__reset test case")
    description = f"{name} (description)"
    images_1 = [test_dataset[0], test_dataset[3], test_dataset[4], test_dataset[5]]
    images_2 = [test_dataset[1], test_dataset[2], test_dataset[3]]
    test_case = TestCase(name, description=description, images=images_1)

    # no op
    with test_case.edit(reset=True) as editor:
        editor.description(description)
    assert test_case.version == 1
    assert test_case.description == description
    assert_test_images_equal(test_case.load_images(), images_1)

    with test_case.edit(reset=True) as editor:
        for image in images_2:
            editor.add(image)

    assert test_case.version == 2
    assert test_case.description == description
    assert_test_images_equal(test_case.load_images(), images_2)


def test__edit__empty(test_dataset: List[TestImage]) -> None:
    test_case = TestCase(with_test_prefix(f"{__file__}::test__edit__empty test case"))

    with test_case.edit() as editor:
        editor.description("description")
    assert test_case.version == 1
    assert test_case.description == "description"

    # add a sample to the test case for later removal
    with test_case.edit() as editor:
        editor.add(test_dataset[0])
    assert test_case.version == 2
    assert len(test_case.load_images()) == 1

    # empty the test case
    with test_case.edit() as editor:
        editor.remove(test_dataset[0])
    assert test_case.version == 3
    assert len(test_case.load_images()) == 0


def test__edit__no_ground_truths(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__no_ground_truths test case")
    test_case = TestCase(name)
    assert test_case.version == 0

    images_no_gt = [image.filter(lambda _: False) for image in test_dataset]
    with test_case.edit() as editor:
        for image in images_no_gt:
            editor.add(image)

    assert_test_images_equal(test_case.load_images(), images_no_gt)


def test__edit__specific_ground_truths(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__edit__specific_ground_truths test case")
    test_case = TestCase(name)

    images_car_only = [image.filter(lambda gt: gt.label == "car") for image in test_dataset]
    images_car_only = [image for image in images_car_only if len(image.ground_truths) > 0]
    with test_case.edit() as editor:
        for image in images_car_only:
            editor.add(image)

    assert test_case.version == 1
    assert_test_images_equal(test_case.load_images(), images_car_only)


def test__edit__no_op() -> None:
    test_case = TestCase(with_test_prefix(f"{__file__}::test__edit__no_op test case"))
    with test_case.edit():
        ...
    assert test_case.version == 0


def test__edit__updated(test_dataset: List[TestImage]) -> None:
    test_case_name = with_test_prefix(f"{__file__} test__edit__updated test case")
    images = [test_dataset[4]]
    test_case = TestCase(test_case_name, images=images)
    assert test_case.version == 1

    # no op
    with test_case.edit() as editor:
        editor.add(images[0])
    assert test_case.version == 1

    updated_image_0 = deepcopy(images[0])
    updated_label = "new label"
    updated_image_0.ground_truths[0].label = updated_label
    loaded_images_before = test_case.load_images()
    # update the existing test sample
    with test_case.edit() as editor:
        editor.add(updated_image_0)
    loaded_images_after = test_case.load_images()
    assert test_case.version == 2
    assert len(loaded_images_before) == len(loaded_images_after) == 1
    assert loaded_images_before != loaded_images_after
    assert updated_label not in [gt.label for gt in loaded_images_before[0].ground_truths]
    assert updated_label in [gt.label for gt in loaded_images_after[0].ground_truths]


def test__update_dataset() -> None:
    dataset = with_test_prefix("test")
    locator = fake_random_locator()
    name_prefix = with_test_prefix(f"{__file__}::test_update_dataset")
    TestCase(f"{name_prefix} test case", images=[TestImage(locator, dataset=dataset)])

    # shouldn't override previously set dataset
    test_case0 = TestCase(f"{name_prefix} test case 0", images=[TestImage(locator)])
    test_case0_images = test_case0.load_images()
    assert len(test_case0_images) == 1
    assert test_case0_images[0].dataset == dataset

    # shouldn't override previously set dataset
    test_case1 = TestCase(f"{name_prefix} test case 1", images=[TestImage(locator, dataset="")])
    test_case1_images = test_case1.load_images()
    assert len(test_case1_images) == 1
    assert test_case1_images[0].dataset == dataset

    # should override
    test_case2 = TestCase(f"{name_prefix} test case 2", images=[TestImage(locator, dataset="new")])
    test_case2_images = test_case2.load_images()
    assert len(test_case2_images) == 1
    assert test_case2_images[0].dataset == "new"


def test__update_metadata() -> None:
    bbox = kolena.detection.metadata.BoundingBox(top_left=(0, 1), bottom_right=(2, 3))
    asset = Asset(locator=fake_random_locator())
    metadata = dict(a="a", b=True, c=3, d=asset, e=bbox)
    test_image0 = TestImage(fake_random_locator(), dataset="test", metadata=metadata)
    TestCase(with_test_prefix(f"{__file__}::test_update_metadata test case 0"), images=[test_image0])

    metadata_updated = {**metadata, **dict(c=4.3, d=False)}
    test_image1 = TestImage(test_image0.locator, dataset=test_image0.dataset, metadata=metadata_updated)
    test_case1 = TestCase(with_test_prefix(f"{__file__}::test_update_metadata test case 1"), images=[test_image1])
    test_case1_images = test_case1.load_images()
    assert len(test_case1_images) == 1
    assert test_case1_images[0].metadata == metadata_updated


def test__create() -> None:
    test_case_name = with_test_prefix(f"{__file__} test__create test case")
    description = f"{test_case_name} (description)"
    test_case = TestCase.create(test_case_name, description)
    assert test_case.version == 0
    assert test_case.name == test_case_name
    assert test_case.description == description
    assert test_case._workflow == WorkflowType.DETECTION


def test__create__with_images(test_dataset: List[TestImage]) -> None:
    name = with_test_prefix(f"{__file__}::test__create__with_images test case")
    description = f"{name} (description)"
    images = [test_dataset[0], test_dataset[3], test_dataset[4]]
    test_case = TestCase.create(name, description, images)
    assert test_case.version == 1
    assert_test_images_equal(test_case.load_images(), images)
    assert test_case._workflow == WorkflowType.DETECTION


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
    ClassificationTestCase(test_case_name)
    with pytest.raises(WorkflowMismatchError) as exc_info:
        TestCase.load(test_case_name)

    exc_info_value = str(exc_info.value)
    assert ClassificationTestCase._workflow.value in exc_info_value
    assert TestCase._workflow.value in exc_info_value


def test__load__with_version_mismatch() -> None:
    test_case_name = with_test_prefix(f"{__file__} test__load__with_version_mismatch test case")
    TestCase(test_case_name)
    mismatch_version = 42
    with pytest.raises(NotFoundError) as exc_info:
        TestCase.load(test_case_name, mismatch_version)

    exc_info_value = str(exc_info.value)
    assert f"(version {mismatch_version})" in exc_info_value


def test__create__with_locator_collision() -> None:
    test_case_name = with_test_prefix(f"{__file__} test__create__with_locator_collision test case")
    locator = fake_random_locator()

    generic_sample = DummyTestSample(  # type: ignore
        locator=locator,
        value=0,
        bbox=GenericBoundingBox(top_left=(0, 0), bottom_right=(0, 0)),
    )
    generic_ground_truth = DummyGroundTruth(label="dummy", value=0)
    GenericTestCase(
        with_test_prefix(f"{__file__}::{test_case_name} generic"),
        test_samples=[
            (
                generic_sample,
                generic_ground_truth,
            ),
        ],
    )
    test_case = TestCase(test_case_name, images=[TestImage(locator)])
    images = test_case.load_images()
    assert len(images) == 1
