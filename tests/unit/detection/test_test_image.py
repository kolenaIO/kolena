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
import uuid
from typing import Any
from typing import Dict

import pytest
from pydantic import ValidationError

import kolena.detection.metadata
from kolena.detection import TestImage
from kolena.detection.ground_truth import BoundingBox
from kolena.detection.ground_truth import ClassificationLabel
from kolena.detection.ground_truth import SegmentationMask
from kolena.detection.metadata import Asset
from kolena.errors import InputValidationError

TEST_LOCATOR = "s3://test-bucket/path/to/image.png"
BBOX_A = BoundingBox("a", (1, 2), (3, 4))
BBOX_B = BoundingBox("b", (3, 4), (5, 6))
SEG_MASK_A = SegmentationMask("a", [(1, 2), (3, 4), (5, 6)])
SEG_MASK_B = SegmentationMask("b", [(2, 2), (3, 3), (4, 4)])


def test__test_image__filter() -> None:
    gt_a = [ClassificationLabel("a"), BBOX_A, SEG_MASK_A]
    gt_other = [
        ClassificationLabel(str(uuid.uuid4())),
        BoundingBox(str(uuid.uuid4()), (1, 2), (3, 4)),
        SegmentationMask(str(uuid.uuid4()), [(1, 2), (3, 4), (5, 6)]),
    ]
    image = TestImage(TEST_LOCATOR, ground_truths=gt_a + gt_other)
    image_filtered = image.filter(lambda gt: gt.label == "a")
    assert image_filtered == TestImage(image.locator, ground_truths=gt_a)

    image_filtered = image.filter(lambda gt: isinstance(gt, ClassificationLabel))
    assert image_filtered == TestImage(image.locator, ground_truths=[gt_a[0], gt_other[0]])


def test__test_image__metadata_types() -> None:
    image = TestImage(TEST_LOCATOR, metadata={"str": "string", "float": 3.14, "int": 99, "bool": True})
    metadata = image.metadata
    assert isinstance(metadata.get("str"), str)
    assert isinstance(metadata.get("float"), float)
    assert isinstance(metadata.get("int"), int)
    assert isinstance(metadata.get("bool"), bool)


def test__test_image__filter__preserves_metadata() -> None:
    metadata = dict(example_bool=True, example_int=1, example_float=2.3, example_asset=Asset("s3://fake/asset.png"))
    image = TestImage(TEST_LOCATOR, ground_truths=[ClassificationLabel("test")], metadata=metadata)
    assert image.filter(lambda gt: gt.label == "fake") == TestImage(image.locator, metadata=image.metadata)


def test__test_image__serde() -> None:
    original = TestImage(
        locator=TEST_LOCATOR,
        dataset="test-dataset",
        ground_truths=[
            ClassificationLabel(str(uuid.uuid4())),
            BoundingBox(str(uuid.uuid4()), (1, 2), (3, 4)),
            SegmentationMask(str(uuid.uuid4()), [(1, 2), (3, 4), (5, 6)]),
        ],
        metadata={
            "none": None,
            "str": "string",
            "float": 1.0,
            "int": 500,
            "bool": False,
            "bbox": kolena.detection.metadata.BoundingBox(top_left=(0, 0), bottom_right=(100, 100)),
            "landmarks": kolena.detection.metadata.Landmarks(points=[(0, 0), (100, 100), (100, 0), (0, 0)]),
            "asset": kolena.detection.metadata.Asset(locator="gs://gs-bucket/path/to/asset.jpg"),
        },
    )

    df = kolena.detection.test_case.TestCase._to_data_frame([original])
    assert len(df) == 1

    recovered = [TestImage._from_record(record) for record in df.itertuples()][0]
    assert original == recovered


@pytest.mark.parametrize(
    "metadata",
    [
        {"list": []},
        {"dict": {}},
    ],
)
def test__test_image__invalid_metadata_types(metadata: Dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        TestImage(locator=TEST_LOCATOR, dataset="test-dataset", ground_truths=[], metadata=metadata)


def test__test_image__nan_metadata() -> None:
    df = kolena.detection.test_case.TestCase._to_data_frame(
        [TestImage(locator=TEST_LOCATOR, ground_truths=[], metadata={"nan": float("nan")})],
    )
    assert len(df) == 1

    recovered = [TestImage._from_record(record) for record in df.itertuples()][0]
    assert recovered.metadata["nan"] is None


@pytest.mark.parametrize(
    "metadata",
    [
        {"inf": float("inf")},
        {"inf": float("-inf")},
    ],
)
def test__test_image__infinite_metadata(metadata: Dict[str, Any]) -> None:
    with pytest.raises(InputValidationError):
        kolena.detection.test_case.TestCase._to_data_frame(
            [TestImage(locator=TEST_LOCATOR, ground_truths=[], metadata=metadata)],
        )


@pytest.mark.parametrize(
    "a,b,expected",
    [
        (TestImage(locator=TEST_LOCATOR), TestImage(locator=TEST_LOCATOR), True),
        (TestImage(locator=TEST_LOCATOR), TestImage(locator="s3://test-bucket/another/image.jpg"), False),
        (TestImage(locator=TEST_LOCATOR, dataset="test"), TestImage(locator=TEST_LOCATOR, dataset="test"), True),
        (TestImage(locator=TEST_LOCATOR, dataset="test"), TestImage(locator=TEST_LOCATOR, dataset="different"), False),
        (
            TestImage(locator=TEST_LOCATOR, ground_truths=[]),
            TestImage(locator=TEST_LOCATOR, ground_truths=[]),
            True,
        ),
        (
            TestImage(locator=TEST_LOCATOR, ground_truths=[BBOX_A, BBOX_B, SEG_MASK_A, SEG_MASK_B]),
            TestImage(locator=TEST_LOCATOR, ground_truths=[BBOX_A, BBOX_B, SEG_MASK_A, SEG_MASK_B]),
            True,
        ),
        (
            # TODO: remove once ground truth ordering is ensured upstream
            TestImage(locator=TEST_LOCATOR, ground_truths=[BBOX_A, BBOX_B, SEG_MASK_A, SEG_MASK_B]),
            TestImage(locator=TEST_LOCATOR, ground_truths=[SEG_MASK_A, BBOX_A, SEG_MASK_B, BBOX_B]),
            True,
        ),
        (
            TestImage(locator=TEST_LOCATOR, ground_truths=[BBOX_A, BBOX_B]),
            TestImage(locator=TEST_LOCATOR, ground_truths=[]),
            False,
        ),
        (
            TestImage(locator=TEST_LOCATOR, ground_truths=[BBOX_A, BBOX_B]),
            TestImage(locator=TEST_LOCATOR, ground_truths=[BBOX_A, BBOX_A]),
            False,
        ),
        (
            TestImage(locator=TEST_LOCATOR, ground_truths=[BBOX_A, BBOX_A]),
            TestImage(locator=TEST_LOCATOR, ground_truths=[BBOX_A, BBOX_A]),
            True,
        ),
        (
            TestImage(locator=TEST_LOCATOR, metadata=dict(a=1, b=True, c="c")),
            TestImage(locator=TEST_LOCATOR, metadata=dict(a=1, b=True, c="c")),
            True,
        ),
        (
            TestImage(locator=TEST_LOCATOR, metadata=dict(a=1, b=True, c="c")),
            TestImage(locator=TEST_LOCATOR, metadata=dict(b=True, c="c", a=1)),
            True,
        ),
        (
            TestImage(locator=TEST_LOCATOR, metadata=dict(a=1, b=True, c="c")),
            TestImage(locator=TEST_LOCATOR, metadata=dict(b=True, c="c")),
            False,
        ),
        (
            TestImage(locator=TEST_LOCATOR, metadata=dict(a=1, b=True, c="c")),
            TestImage(locator=TEST_LOCATOR),
            False,
        ),
    ],
)
def test__test_image__equality(a: TestImage, b: TestImage, expected: bool) -> None:
    assert (a == b) is expected
