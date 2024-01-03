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
import random
from typing import List
from typing import Tuple

from kolena.detection import ground_truth
from kolena.detection import inference
from kolena.detection._internal import BaseTestImage

fake_labels = [
    "car",
    "bike",
    "house",
    "airplane",
    "boat",
    "bus",
    "animal",
    "person",
    "cow",
    "cat",
    "dog",
    "parakeet",
    "weasel",
    "rabbit",
    "mouse",
    "rat",
    "anteater",
    "aardvark",
    "whale",
    "seal",
    "walrus",
    "butterfly",
    "hawk",
    "pigeon",
    "goose",
]


def fake_label() -> str:
    return random.choice(fake_labels)


def fake_points(n: int) -> List[Tuple[float, float]]:
    return [(round(random.random() * 300, 3), round(random.random(), 3)) for _ in range(n)]


def fake_gt_classification_label() -> ground_truth.ClassificationLabel:
    return ground_truth.ClassificationLabel(fake_label())


def fake_gt_bounding_box() -> ground_truth.BoundingBox:
    return ground_truth.BoundingBox(fake_label(), *fake_points(2))


def fake_gt_segmentation_mask() -> ground_truth.SegmentationMask:
    return ground_truth.SegmentationMask(fake_label(), fake_points(random.randint(3, 15)))


def fake_confidence() -> float:
    return round(random.random(), 3)


def fake_inference_classification_label() -> inference.ClassificationLabel:
    return inference.ClassificationLabel(fake_label(), fake_confidence())


def fake_inference_bounding_box() -> inference.BoundingBox:
    return inference.BoundingBox(fake_label(), fake_confidence(), *fake_points(2))


def fake_inference_segmentation_mask() -> inference.SegmentationMask:
    return inference.SegmentationMask(fake_label(), fake_confidence(), fake_points(random.randint(3, 15)))


def assert_test_images_equal(actual: List[BaseTestImage], expected: List[BaseTestImage]) -> None:
    assert len(actual) == len(expected)
    actual = sorted(actual, key=lambda x: x.locator)
    expected = sorted(expected, key=lambda x: x.locator)
    for a, b in zip(actual, expected):
        assert a == b
