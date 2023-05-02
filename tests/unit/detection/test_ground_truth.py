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
import pytest

from kolena.detection.ground_truth import BoundingBox
from kolena.detection.ground_truth import ClassificationLabel
from kolena.detection.ground_truth import SegmentationMask


def test_ground_truth__classification_label() -> None:
    test_label = "test-label"
    gt = ClassificationLabel(test_label)
    assert gt.label == test_label

    assert gt._to_dict() == dict(data_type="CLASSIFICATION_LABEL", data_object=dict(label=test_label, difficult=False))


def test_ground_truth__bounding_box() -> None:
    test_label = "test-label"
    test_point1 = (0.1234, 100.213)
    test_point2 = (1.1234, 101.213)
    gt = BoundingBox(test_label, test_point1, test_point2)
    assert gt.label == test_label
    assert gt.top_left == test_point1
    assert gt.bottom_right == test_point2

    assert gt._to_dict() == dict(
        data_type="BOUNDING_BOX",
        data_object=dict(label=test_label, points=[test_point1, test_point2], difficult=False),
    )


def test_ground_truth__segmentation_mask() -> None:
    test_label = "test-label"
    test_points = [(0.1234, 100.213), (1.1234, 101.213), (2.1234, 110.213)]

    with pytest.raises(ValueError):
        SegmentationMask(test_label, test_points[:2])

    gt = SegmentationMask(test_label, test_points)
    assert gt.label == test_label
    assert gt.points == test_points

    assert gt._to_dict() == dict(
        data_type="SEGMENTATION_MASK",
        data_object=dict(label=test_label, points=test_points, difficult=False),
    )
