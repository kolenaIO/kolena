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


def test__test_image__filter() -> None:
    gt_a = [
        ClassificationLabel("a"),
        BoundingBox("a", (1, 2), (3, 4)),
        SegmentationMask("a", [(1, 2), (3, 4), (5, 6)]),
    ]
    gt_other = [
        ClassificationLabel(str(uuid.uuid4())),
        BoundingBox(str(uuid.uuid4()), (1, 2), (3, 4)),
        SegmentationMask(str(uuid.uuid4()), [(1, 2), (3, 4), (5, 6)]),
    ]
    image = TestImage("s3://fake/locator.png", ground_truths=gt_a + gt_other)
    image_filtered = image.filter(lambda gt: gt.label == "a")
    assert image_filtered == TestImage(image.locator, ground_truths=gt_a)

    image_filtered = image.filter(lambda gt: isinstance(gt, ClassificationLabel))
    assert image_filtered == TestImage(image.locator, ground_truths=[gt_a[0], gt_other[0]])


def test__test_image__metadata_types() -> None:
    image = TestImage("s3://fake/locator.png", metadata={"str": "string", "float": 3.14, "int": 99, "bool": True})
    metadata = image.metadata
    assert isinstance(metadata.get("str"), str)
    assert isinstance(metadata.get("float"), float)
    assert isinstance(metadata.get("int"), int)
    assert isinstance(metadata.get("bool"), bool)


def test__test_image__filter__preserves_metadata() -> None:
    metadata = dict(example_bool=True, example_int=1, example_float=2.3, example_asset=Asset("s3://fake/asset.png"))
    image = TestImage("s3://fake/locator.png", ground_truths=[ClassificationLabel("test")], metadata=metadata)
    assert image.filter(lambda gt: gt.label == "fake") == TestImage(image.locator, metadata=image.metadata)


def test__test_image__serde() -> None:
    original = TestImage(
        locator="s3://test-bucket/path/to/file.png",
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
        TestImage(
            locator="s3://test-bucket/path/to/file.png",
            dataset="test-dataset",
            ground_truths=[],
            metadata=metadata,
        )


def test__test_image__nan_metadata() -> None:
    df = kolena.detection.test_case.TestCase._to_data_frame(
        [
            TestImage(
                locator="s3://test-bucket/path/to/file.png",
                ground_truths=[],
                metadata={"nan": float("nan")},
            ),
        ],
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
            [
                TestImage(
                    locator="s3://test-bucket/path/to/file.png",
                    ground_truths=[],
                    metadata=metadata,
                ),
            ],
        )
