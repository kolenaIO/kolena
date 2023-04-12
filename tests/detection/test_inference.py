from random import random

import pytest

from kolena.detection.inference import BoundingBox
from kolena.detection.inference import ClassificationLabel
from kolena.detection.inference import SegmentationMask


def test_inference__classification_label() -> None:
    test_label = "test-label"
    confidence = random()
    inf = ClassificationLabel(test_label, confidence)
    assert inf.label == test_label
    assert inf.confidence == confidence


def test_inference__bounding_box() -> None:
    test_label = "test-label"
    confidence = random()
    test_point1 = (0.1234, 100.213)
    test_point2 = (1.1234, 101.213)
    inf = BoundingBox(test_label, confidence, test_point1, test_point2)
    assert inf.label == test_label
    assert inf.confidence == confidence
    assert inf.top_left == test_point1
    assert inf.bottom_right == test_point2


def test_inference__segmentation_mask() -> None:
    test_label = "test-label"
    confidence = random()
    test_point1 = (0.1234, 100.213)
    test_point2 = (1.1234, 101.213)
    test_point3 = (2.1234, 110.213)
    # TODO (andrew): Should we validate that the polygon created is a real polygon?

    with pytest.raises(ValueError):
        SegmentationMask(test_label, confidence, [test_point1, test_point2])

    inf = SegmentationMask(test_label, confidence, [test_point1, test_point2, test_point3])
    assert inf.label == test_label
    assert inf.confidence == confidence
    assert inf.points == [test_point1, test_point2, test_point3]
