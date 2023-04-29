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

from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Polygon
from kolena.workflow.metrics._geometry import iou


@pytest.mark.parametrize(
    "box1, box2, expected_iou",
    [
        # Boxes do not intersect
        (BoundingBox((0, 0), (0, 0)), BoundingBox((0, 0), (0, 0)), 0.0),
        (BoundingBox((0.5, 0.5), (1.5, 1.5)), BoundingBox((4.5, 4.5), (5.5, 5.5)), 0.0),
        (BoundingBox((0, 0), (1, 1)), BoundingBox((1, 1), (2, 2)), 0.0),
        (BoundingBox((0, 1), (1, 2)), BoundingBox((1, 1), (2, 2)), 0.0),
        (BoundingBox((0, 2), (1, 3)), BoundingBox((1, 1), (2, 2)), 0.0),
        (BoundingBox((1, 0), (2, 1)), BoundingBox((1, 1), (2, 2)), 0.0),
        (BoundingBox((2, 0), (3, 1)), BoundingBox((1, 1), (2, 2)), 0.0),
        (BoundingBox((2, 1), (3, 2)), BoundingBox((1, 1), (2, 2)), 0.0),
        (BoundingBox((2, 2), (3, 3)), BoundingBox((1, 1), (2, 2)), 0.0),
        (BoundingBox((1, 2), (2, 3)), BoundingBox((1, 1), (2, 2)), 0.0),
        # One box contained within the other
        (BoundingBox((0, 0), (5, 5)), BoundingBox((0, 0), (1, 1)), 0.04),
        (BoundingBox((0, 0), (5, 5)), BoundingBox((0, 4), (1, 5)), 0.04),
        (BoundingBox((0, 0), (5, 5)), BoundingBox((4, 0), (5, 1)), 0.04),
        (BoundingBox((0, 0), (5, 5)), BoundingBox((4, 4), (5, 5)), 0.04),
        (BoundingBox((0, 0), (5, 5)), BoundingBox((1, 1), (2, 2)), 0.04),
        # Partial overlap
        (BoundingBox((0, 0), (2, 2)), BoundingBox((1, 1), (3, 3)), 1 / 7),
        (BoundingBox((1, 1), (3, 3)), BoundingBox((0, 0), (2, 2)), 1 / 7),
        (BoundingBox((0, 0), (2, 2)), BoundingBox((1, 1), (3, 3)), 1 / 7),
        (BoundingBox((1, 1), (3, 3)), BoundingBox((0, 0), (2, 2)), 1 / 7),
        (BoundingBox((0, 0), (2, 2)), BoundingBox((1, 1), (3, 3)), 1 / 7),
        (BoundingBox((1, 1), (3, 3)), BoundingBox((0, 0), (2, 2)), 1 / 7),
        # Complete overlap
        (BoundingBox((0, 0), (2, 2)), BoundingBox((0, 0), (2, 2)), 1.0),
        (BoundingBox((0.1, 0.1), (2, 2)), BoundingBox((0.1, 0.1), (2, 2)), 1.0),
        (BoundingBox((1, 3.14), (101.1, 54.321)), BoundingBox((1, 3.14), (101.1, 54.321)), 1.0),
        # Extra
        (BoundingBox((1, 1), (2, 2)), BoundingBox((1, 0), (2, 1)), 0.0),
        (BoundingBox((1, 1), (2, 2)), BoundingBox((2, 0), (3, 1)), 0.0),
        (BoundingBox((4, 4), (5, 5)), BoundingBox((0, 0), (5, 5)), 0.04),
        (BoundingBox((1, 1), (2, 2)), BoundingBox((0, 0), (5, 5)), 0.04),
        (BoundingBox((0, 0), (4, 4)), BoundingBox((2, 2), (6, 6)), 4 / 28),
        (BoundingBox((2, 2), (6, 6)), BoundingBox((0, 0), (4, 4)), 4 / 28),
        (BoundingBox((0, 0), (4, 3)), BoundingBox((2, 1), (5, 4)), 4 / 17),
        (BoundingBox((2, 1), (5, 4)), BoundingBox((0, 0), (4, 3)), 4 / 17),
        (BoundingBox((0, 0), (3, 4)), BoundingBox((1, 2), (4, 6)), 0.2),
        (BoundingBox((1, 2), (4, 6)), BoundingBox((0, 0), (3, 4)), 0.2),
    ],
)
def test_iou_bbox(box1: BoundingBox, box2: BoundingBox, expected_iou: float) -> None:
    assert iou(box1, box2) == pytest.approx(expected_iou, abs=1e-5)


@pytest.mark.parametrize(
    "points1, points2, expected_iou",
    [
        # Non-overlapping polygons
        (Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(2, 2), (3, 2), (3, 3), (2, 3)]), 0.0),
        # Identical polygons
        (Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), Polygon([(0, 0), (1, 0), (1, 1), (0, 1)]), 1.0),
        (
            Polygon([(2.3, 4.5), (6.7, 4.5), (10.1, 12.32), (1.54, 6.65)]),
            Polygon([(2.3, 4.5), (6.7, 4.5), (10.1, 12.32), (1.54, 6.65)]),
            1.0,
        ),
        # Partial overlap
        (Polygon([(0, 0), (2, 0), (2, 2), (0, 2)]), Polygon([(1, 1), (3, 1), (3, 3), (1, 3)]), 1 / 7),
        (Polygon([(0, 0), (10, 0), (10, 10)]), Polygon([(0, 10), (10, 10), (10, 0)]), 1 / 3),
        (Polygon([(0, 0), (5, 0), (5, -100), (6, 0), (10, 0), (10, 10)]), Polygon([(0, 10), (10, 10), (10, 0)]), 0.2),
        # Inside another
        (Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]), Polygon([(1, 1), (2, 1), (2, 2), (1, 2)]), 1 / 9),
        # Mixed types
        (Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]), BoundingBox((1, 1), (2, 2)), 1 / 9),
        (BoundingBox((1, 1), (2, 2)), Polygon([(0, 0), (3, 0), (3, 3), (0, 3)]), 1 / 9),
        (BoundingBox((0, 0), (4, 4)), BoundingBox((2, 2), (6, 6)), 4 / 28),
    ],
)
def test_iou(points1, points2, expected_iou):
    iou_value = iou(points1, points2)
    assert iou_value == pytest.approx(expected_iou, abs=1e-5)
