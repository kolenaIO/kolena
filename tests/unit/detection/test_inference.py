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
from random import random

import pytest

from kolena.detection.inference import BoundingBox
from kolena.detection.inference import ClassificationLabel
from kolena.detection.inference import SegmentationMask


def test__inference__classification_label() -> None:
    test_label = "test-label"
    confidence = random()
    inf = ClassificationLabel(test_label, confidence)
    assert inf.label == test_label
    assert inf.confidence == confidence


def test__inference__bounding_box() -> None:
    test_label = "test-label"
    confidence = random()
    test_point1 = (0.1234, 100.213)
    test_point2 = (1.1234, 101.213)
    inf = BoundingBox(test_label, confidence, test_point1, test_point2)
    assert inf.label == test_label
    assert inf.confidence == confidence
    assert inf.top_left == test_point1
    assert inf.bottom_right == test_point2


def test__inference__segmentation_mask() -> None:
    test_label = "test-label"
    confidence = random()
    test_point1 = (0.1234, 100.213)
    test_point2 = (1.1234, 101.213)
    test_point3 = (2.1234, 110.213)
    # TODO: Should we validate that the polygon created is a real polygon?

    with pytest.raises(ValueError):
        SegmentationMask(test_label, confidence, [test_point1, test_point2])

    inf = SegmentationMask(test_label, confidence, [test_point1, test_point2, test_point3])
    assert inf.label == test_label
    assert inf.confidence == confidence
    assert inf.points == [test_point1, test_point2, test_point3]
