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
