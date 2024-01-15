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
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pytest

from kolena.errors import InputValidationError
from kolena.metrics import iou
from kolena.metrics import match_inferences
from kolena.metrics import match_inferences_multiclass
from kolena.metrics._geometry import GT
from kolena.metrics._geometry import Inf
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import Polygon
from kolena.workflow.annotation import ScoredBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledPolygon


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
def test__iou__bbox(box1: BoundingBox, box2: BoundingBox, expected_iou: float) -> None:
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
def test__iou(points1: Union[BoundingBox, Polygon], points2: Union[BoundingBox, Polygon], expected_iou: float) -> None:
    iou_value = iou(points1, points2)
    assert iou_value == pytest.approx(expected_iou, abs=1e-5)


@pytest.mark.parametrize(
    "test_name, ground_truths, inferences, ignored_ground_truths, "
    + "expected_matched, expected_unmatched_gt, expected_unmatched_inf",
    [
        (
            # test_name
            "1. Single inference below IOU threshold => no match",
            # ground_truth
            [BoundingBox(top_left=(10, 10), bottom_right=(60, 60))],
            # inference
            [ScoredBoundingBox(score=0.1, top_left=(1, 1), bottom_right=(6, 6))],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [BoundingBox(top_left=(10, 10), bottom_right=(60, 60))],
            # expected_unmatched_inf
            [ScoredBoundingBox(score=0.1, top_left=(1, 1), bottom_right=(6, 6))],
        ),
        (
            # test_name
            "2. Single inference above IOU threshold, matching GT => match",
            # ground_truth
            [BoundingBox(top_left=(1, 1), bottom_right=(6, 6))],
            # inference
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 1), bottom_right=(6, 6))],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(1, 1), bottom_right=(6, 6)),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 1), bottom_right=(6, 6)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "3. Single inference below IOU threshold, high score => no match",
            # ground_truth
            [BoundingBox(top_left=(1, 1), bottom_right=(6, 6))],
            # inference
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(5, 5), bottom_right=(10, 10))],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [BoundingBox(top_left=(1, 1), bottom_right=(6, 6))],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(5, 5), bottom_right=(10, 10))],
        ),
        (
            # test_name
            "4. Single inference with GT with IOU = 0 => no match",
            # ground_truth
            [BoundingBox(top_left=(1, 1), bottom_right=(6, 6))],
            # inference
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(7, 7), bottom_right=(10, 10))],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [BoundingBox(top_left=(1, 1), bottom_right=(6, 6))],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(7, 7), bottom_right=(10, 10))],
        ),
        (
            # test_name
            "5. Single inference, no GTs => no match",
            # ground_truth
            [],
            # inference
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(7, 7), bottom_right=(10, 10))],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(7, 7), bottom_right=(10, 10))],
        ),
        (
            # test_name
            "5.5 No inference, one GT => no match",
            # ground_truth
            [BoundingBox(top_left=(1, 1), bottom_right=(6, 6))],
            # inference
            [],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [BoundingBox(top_left=(1, 1), bottom_right=(6, 6))],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "6. No inferences, no GTs => no match",
            # ground_truth
            [],
            # inference
            [],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "7. Two inferences both matching GT with same IOU => higher confidence inference matched",
            # ground_truth
            [BoundingBox(top_left=(4, 4), bottom_right=(8, 8))],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(4, 4), bottom_right=(9, 9)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(4, 4), bottom_right=(8, 8)),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(4, 4), bottom_right=(9, 9)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8))],
        ),
        (
            # test_name
            "8. 2 inf, both matching gt, higher conf has lower IOU => higher conf inf matched",
            # ground_truth
            [BoundingBox(top_left=(4, 4), bottom_right=(8, 8))],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(4, 4), bottom_right=(9.1, 9.1)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(4, 4), bottom_right=(8, 8)),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(4, 4), bottom_right=(9.1, 9.1)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8))],
        ),
        (
            # test_name
            "9. 2 inf, both matching gt, higher conf has higher IOU => higher conf inf matched",
            # ground_truth
            [BoundingBox(top_left=(4, 4), bottom_right=(8, 8))],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ScoredLabeledBoundingBox(score=0.7, label="cow", top_left=(4, 4), bottom_right=(9.1, 9.1)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(4, 4), bottom_right=(8, 8)),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.7, label="cow", top_left=(4, 4), bottom_right=(9.1, 9.1))],
        ),
        (
            # test_name
            "10. Both inferences match to a GT => higher confidence inference matched",
            # ground_truth
            [BoundingBox(top_left=(4, 4), bottom_right=(8, 8))],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.4, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(4, 4), bottom_right=(8, 8)),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.4, label="cow", top_left=(3, 3), bottom_right=(8, 8))],
        ),
        (
            # test_name
            "10.1 Both inferences match to a GT => higher confidence inference matched, order swap",
            # ground_truth
            [BoundingBox(top_left=(4, 4), bottom_right=(8, 8))],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ScoredLabeledBoundingBox(score=0.4, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(4, 4), bottom_right=(8, 8)),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.4, label="cow", top_left=(3, 3), bottom_right=(8, 8))],
        ),
        (
            # test_name
            "11. Single inference, two GT with IOU > T => match with higher IOU GT despite out of order",
            # ground_truth
            [BoundingBox(top_left=(3, 3), bottom_right=(8, 8)), BoundingBox(top_left=(3, 3), bottom_right=(9, 9))],
            # inference
            [ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(9, 9))],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(3, 3), bottom_right=(9, 9)),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(9, 9)),
                ),
            ],
            # expected_unmatched_gt
            [BoundingBox(top_left=(3, 3), bottom_right=(8, 8))],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "11.1 Single inference, two GT with IOU > T => match with higher IOU GT in order",
            # ground_truth
            [BoundingBox(top_left=(3, 3), bottom_right=(9, 9)), BoundingBox(top_left=(3, 3), bottom_right=(8, 8))],
            # inference
            [ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(9, 9))],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(3, 3), bottom_right=(9, 9)),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(9, 9)),
                ),
            ],
            # expected_unmatched_gt
            [BoundingBox(top_left=(3, 3), bottom_right=(8, 8))],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "12. Two inferences A, B, two GT X, Y, A.score = 0.9, B.score = 0.6",
            # Y = BoundingBox((1,5), (15,16))
            # X = BoundingBox((1,12), (15,27))
            # A = BoundingBox((1,5), (15,27))
            # B = BoundingBox((1,17), (15,27))
            # ground_truth
            [BoundingBox(top_left=(1, 5), bottom_right=(15, 16)), BoundingBox(top_left=(1, 12), bottom_right=(15, 27))],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.6, label="cow", top_left=(1, 17), bottom_right=(15, 27)),
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(1, 12), bottom_right=(15, 27)),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
                ),
            ],
            # expected_unmatched_gt
            [BoundingBox(top_left=(1, 5), bottom_right=(15, 16))],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.6, label="cow", top_left=(1, 17), bottom_right=(15, 27))],
        ),
        (
            # test_name
            "12.1 Two inferences A, B, two GT X, Y, A.score = 0.9, B.score = 0.6",
            # Y = BoundingBox((1,5), (15,16))
            # X = BoundingBox((1,12), (15,27))
            # A = BoundingBox((1,5), (15,27))
            # B = BoundingBox((1,17), (15,27))
            # ground_truth
            [BoundingBox(top_left=(1, 12), bottom_right=(15, 27)), BoundingBox(top_left=(1, 5), bottom_right=(15, 16))],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.6, label="cow", top_left=(1, 17), bottom_right=(15, 27)),
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(1, 12), bottom_right=(15, 27)),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
                ),
            ],
            # expected_unmatched_gt
            [BoundingBox(top_left=(1, 5), bottom_right=(15, 16))],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.6, label="cow", top_left=(1, 17), bottom_right=(15, 27))],
        ),
        (
            # test_name
            "13. Two inferences A, B, two GT X, Y, A.score = 1, B.score = 0.5",
            # Y = BoundingBox((1,5), (15,16))
            # X = BoundingBox((1,12), (15,27))
            # A = BoundingBox((1,5), (15,27))
            # B = BoundingBox((1,12), (15,22))
            # ground_truth
            [BoundingBox(top_left=(1, 12), bottom_right=(15, 27)), BoundingBox(top_left=(1, 5), bottom_right=(15, 16))],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(1, 12), bottom_right=(15, 22)),
                ScoredLabeledBoundingBox(score=1.0, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(1, 12), bottom_right=(15, 27)),
                    ScoredLabeledBoundingBox(score=1.0, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
                ),
            ],
            # expected_unmatched_gt
            [BoundingBox(top_left=(1, 5), bottom_right=(15, 16))],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 22), label="cow", score=0.5)],
        ),
        (
            # test_name
            "13.1 Two inferences A, B, two GT X, Y, A.score = 1, B.score = 0.5",
            # Y = BoundingBox((1,5), (15,16))
            # X = BoundingBox((1,12), (15,27))
            # A = BoundingBox((1,5), (15,27))
            # B = BoundingBox((1,12), (15,22))
            # ground_truth
            [BoundingBox(top_left=(1, 5), bottom_right=(15, 16)), BoundingBox(top_left=(1, 12), bottom_right=(15, 27))],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(1, 12), bottom_right=(15, 22)),
                ScoredLabeledBoundingBox(score=1.0, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    BoundingBox(top_left=(1, 12), bottom_right=(15, 27)),
                    ScoredLabeledBoundingBox(score=1.0, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
                ),
            ],
            # expected_unmatched_gt
            [BoundingBox(top_left=(1, 5), bottom_right=(15, 16))],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 22), label="cow", score=0.5)],
        ),
        (
            # test_name
            "14. Ignored gt has perfect match",
            # ground_truth
            [LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110))],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(label="cow", top_left=(99, 99), bottom_right=(112, 112))],
            # expected_matched
            [],
            # expected_unmatched_gt
            [LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110))],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "16. Single inference, two GT with both IOU > T",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110)),
                LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(111, 111)),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [
                (
                    LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(111, 111)),
                    ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
                ),
            ],
            # expected_unmatched_gt
            [LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110))],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "16.1 Single inference, two GT with one IOU > T and one < T",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110)),
                LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(111, 111)),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [
                (
                    LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110)),
                    ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
                ),
            ],
            # expected_unmatched_gt
            [LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(111, 111))],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "17. Single inference, two GT with both IOU < T",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(110, 110)),
                LabeledBoundingBox(label="cow", top_left=(5, 5), bottom_right=(111, 111)),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(110, 110)),
                LabeledBoundingBox(label="cow", top_left=(5, 5), bottom_right=(111, 111)),
            ],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112))],
        ),
        (
            # test_name
            "18. 3 infs, two infs can match with one ignored gt",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(2, 2), bottom_right=(11, 11)),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(2, 2), bottom_right=(11, 11)),
                ScoredLabeledBoundingBox(score=0.7, label="cow", top_left=(10, 10), bottom_right=(11, 11)),
            ],
            # ignored_ground_truths
            [
                LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                LabeledBoundingBox(label="cow", top_left=(2, 2), bottom_right=(11, 11)),
            ],
            # expected_matched
            [
                (
                    LabeledBoundingBox(label="cow", top_left=(2, 2), bottom_right=(11, 11)),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(2, 2), bottom_right=(11, 11)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.7, label="cow", top_left=(10, 10), bottom_right=(11, 11))],
        ),
        (
            # test_name
            "18.1 3 infs, two infs can match with one ignored gt",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                LabeledBoundingBox(label="cow", top_left=(11, 11), bottom_right=(12, 12)),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(2, 2), bottom_right=(11, 11)),
                ScoredLabeledBoundingBox(score=0.7, label="cow", top_left=(2, 3), bottom_right=(11, 11)),
            ],
            # ignored_ground_truths
            [
                LabeledBoundingBox(label="cow", top_left=(2, 2), bottom_right=(11, 11)),
            ],
            # expected_matched
            [
                (
                    LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                ),
            ],
            # expected_unmatched_gt
            [LabeledBoundingBox(label="cow", top_left=(11, 11), bottom_right=(12, 12))],
            # expected_unmatched_inf
            [],
        ),
    ],
)
def test__match_inferences(
    test_name: str,
    ground_truths: List[GT],
    inferences: List[Inf],
    ignored_ground_truths: Optional[List[GT]],
    expected_matched: List[Tuple[GT, Inf]],
    expected_unmatched_gt: List[GT],
    expected_unmatched_inf: List[Inf],
) -> None:
    matches = match_inferences(
        ground_truths,
        inferences,
        ignored_ground_truths=ignored_ground_truths,
    )

    assert expected_matched == matches.matched
    assert expected_unmatched_gt == matches.unmatched_gt
    assert expected_unmatched_inf == matches.unmatched_inf


def test__match_inferences__invalid_mode() -> None:
    with pytest.raises(InputValidationError):
        match_inferences(
            [BoundingBox(top_left=(100, 100), bottom_right=(110, 110))],
            [ScoredBoundingBox(score=0.5, top_left=(99, 99), bottom_right=(112, 112))],
            mode="not pascal",
        )


@pytest.mark.parametrize(
    "test_name, ground_truths, inferences, ignored_ground_truths, "
    + "expected_matched, expected_unmatched_gt, expected_unmatched_inf",
    [
        (
            # test_name
            "02. Single inference above IOU threshold, matching GT => match",
            # ground_truth
            [LabeledBoundingBox(top_left=(1, 1), bottom_right=(6, 6), label="cow")],
            # inference
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 1), bottom_right=(6, 6))],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(1, 1), bottom_right=(6, 6), label="cow"),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 1), bottom_right=(6, 6)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "03. Single inference below IOU threshold, high score => no match",
            # ground_truth
            [LabeledBoundingBox(top_left=(1, 1), bottom_right=(6, 6), label="cow")],
            # inference
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(5, 5), bottom_right=(10, 10))],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(1, 1), bottom_right=(6, 6), label="cow"), None)],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(5, 5), bottom_right=(10, 10))],
        ),
        (
            # test_name
            "04. Single inference with GT with IOU = 0 => no match",
            # ground_truth
            [LabeledBoundingBox(top_left=(1, 1), bottom_right=(6, 6), label="cow")],
            # inference
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(7, 7), bottom_right=(10, 10))],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(1, 1), bottom_right=(6, 6), label="cow"), None)],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(7, 7), bottom_right=(10, 10))],
        ),
        (
            # test_name
            "05. Single inference, no GTs => no match",
            # ground_truth
            [],
            # inference
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(7, 7), bottom_right=(10, 10))],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(7, 7), bottom_right=(10, 10))],
        ),
        (
            # test_name
            "05.5 No inference, one GT => no match",
            # ground_truth
            [LabeledBoundingBox(top_left=(1, 1), bottom_right=(6, 6), label="cow")],
            # inference
            [],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(1, 1), bottom_right=(6, 6), label="cow"), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "06. No inferences, no GTs => no match",
            # ground_truth
            [],
            # inference
            [],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "07. Two inferences both matching GT with same IOU => higher confidence inference matched",
            # ground_truth
            [LabeledBoundingBox(top_left=(4, 4), bottom_right=(8, 8), label="cow")],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(4, 4), bottom_right=(9, 9)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(4, 4), bottom_right=(8, 8), label="cow"),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(4, 4), bottom_right=(9, 9)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8))],
        ),
        (
            # test_name
            "08. 2 inf, both matching gt, higher conf has lower IOU => higher conf inf matched",
            # ground_truth
            [LabeledBoundingBox(top_left=(4, 4), bottom_right=(8, 8), label="cow")],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(4, 4), bottom_right=(9.1, 9.1)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(4, 4), bottom_right=(8, 8), label="cow"),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(4, 4), bottom_right=(9.1, 9.1)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8))],
        ),
        (
            # test_name
            "09. 2 inf, both matching gt, higher conf has higher IOU => higher conf inf matched",
            # ground_truth
            [LabeledBoundingBox(top_left=(4, 4), bottom_right=(8, 8), label="cow")],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ScoredLabeledBoundingBox(score=0.7, label="cow", top_left=(4, 4), bottom_right=(9.1, 9.1)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(4, 4), bottom_right=(8, 8), label="cow"),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.7, label="cow", top_left=(4, 4), bottom_right=(9.1, 9.1))],
        ),
        (
            # test_name
            "010. Both inferences match to a GT => higher confidence inference matched",
            # ground_truth
            [LabeledBoundingBox(top_left=(4, 4), bottom_right=(8, 8), label="cow")],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.4, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(4, 4), bottom_right=(8, 8), label="cow"),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.4, label="cow", top_left=(3, 3), bottom_right=(8, 8))],
        ),
        (
            # test_name
            "010.1 Both inferences match to a GT => higher confidence inference matched, order swap",
            # ground_truth
            [LabeledBoundingBox(top_left=(4, 4), bottom_right=(8, 8), label="cow")],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ScoredLabeledBoundingBox(score=0.4, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(4, 4), bottom_right=(8, 8), label="cow"),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(8, 8)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.4, label="cow", top_left=(3, 3), bottom_right=(8, 8))],
        ),
        (
            # test_name
            "011. Single inference, two GT with IOU > T => match with higher IOU GT despite out of order",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(8, 8), label="cow"),
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(9, 9), label="cow"),
            ],
            # inference
            [ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(9, 9))],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(3, 3), bottom_right=(9, 9), label="cow"),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(9, 9)),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(3, 3), bottom_right=(8, 8), label="cow"), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "011.1 Single inference, two GT with IOU > T => match with higher IOU GT in order, order swap",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(9, 9), label="cow"),
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(8, 8), label="cow"),
            ],
            # inference
            [ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(9, 9))],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(3, 3), bottom_right=(9, 9), label="cow"),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(3, 3), bottom_right=(9, 9)),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(3, 3), bottom_right=(8, 8), label="cow"), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "012. Two inferences A, B, two GT X, Y, A.score = 0.9, B.score = 0.6",
            # Y = LabeledBoundingBox((1,5), (15,16))
            # X = LabeledBoundingBox((1,12), (15,27))
            # A = LabeledBoundingBox((1,5), (15,27))
            # B = LabeledBoundingBox((1,17), (15,27))
            # ground_truth
            [
                LabeledBoundingBox(top_left=(1, 5), bottom_right=(15, 16), label="cow"),
                LabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 27), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.6, label="cow", top_left=(1, 17), bottom_right=(15, 27)),
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 27), label="cow"),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(1, 5), bottom_right=(15, 16), label="cow"), None)],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.6, label="cow", top_left=(1, 17), bottom_right=(15, 27))],
        ),
        (
            # test_name
            "012.1 Two inferences A, B, two GT X, Y, A.score = 0.9, B.score = 0.6, order swap",
            # Y = LabeledBoundingBox((1,5), (15,16))
            # X = LabeledBoundingBox((1,12), (15,27))
            # A = LabeledBoundingBox((1,5), (15,27))
            # B = LabeledBoundingBox((1,17), (15,27))
            # ground_truth
            [
                LabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 27), label="cow"),
                LabeledBoundingBox(top_left=(1, 5), bottom_right=(15, 16), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.6, label="cow", top_left=(1, 17), bottom_right=(15, 27)),
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 27), label="cow"),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(1, 5), bottom_right=(15, 16), label="cow"), None)],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.6, label="cow", top_left=(1, 17), bottom_right=(15, 27))],
        ),
        (
            # test_name
            "013. Two inferences A, B, two GT X, Y, A.score = 1, B.score = 0.5",
            # Y = LabeledBoundingBox((1,5), (15,16))
            # X = LabeledBoundingBox((1,12), (15,27))
            # A = LabeledBoundingBox((1,5), (15,27))
            # B = LabeledBoundingBox((1,12), (15,22))
            # ground_truth
            [
                LabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 27), label="cow"),
                LabeledBoundingBox(top_left=(1, 5), bottom_right=(15, 16), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(1, 12), bottom_right=(15, 22)),
                ScoredLabeledBoundingBox(score=1.0, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 27), label="cow"),
                    ScoredLabeledBoundingBox(score=1.0, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(1, 5), bottom_right=(15, 16), label="cow"), None)],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 22), label="cow", score=0.5)],
        ),
        (
            # test_name
            "013.1 Two inferences A, B, two GT X, Y, A.score = 1, B.score = 0.5, order swap",
            # Y = LabeledBoundingBox((1,5), (15,16))
            # X = LabeledBoundingBox((1,12), (15,27))
            # A = LabeledBoundingBox((1,5), (15,27))
            # B = LabeledBoundingBox((1,12), (15,22))
            # ground_truth
            [
                LabeledBoundingBox(top_left=(1, 5), bottom_right=(15, 16), label="cow"),
                LabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 27), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(1, 12), bottom_right=(15, 22)),
                ScoredLabeledBoundingBox(score=1.0, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 27), label="cow"),
                    ScoredLabeledBoundingBox(score=1.0, label="cow", top_left=(1, 5), bottom_right=(15, 27)),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(1, 5), bottom_right=(15, 16), label="cow"), None)],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 12), bottom_right=(15, 22), label="cow", score=0.5)],
        ),
        (
            # test_name
            "014. Ignored gt has perfect match",
            # ground_truth
            [LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110))],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(label="cow", top_left=(99, 99), bottom_right=(112, 112))],
            # expected_matched
            [],
            # expected_unmatched_gt
            [(LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110)), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "016. Single inference, two GT with both IOU > T",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110)),
                LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(111, 111)),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [
                (
                    LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(111, 111)),
                    ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110)), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "016.1 Single inference, two GT with one IOU > T and one < T, order swap",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110)),
                LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(111, 111)),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [
                (
                    LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110)),
                    ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(111, 111)), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "017. Single inference, two GT with both IOU < T",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(110, 110)),
                LabeledBoundingBox(label="cow", top_left=(5, 5), bottom_right=(111, 111)),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112)),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(110, 110)), None),
                (LabeledBoundingBox(label="cow", top_left=(5, 5), bottom_right=(111, 111)), None),
            ],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112))],
        ),
        (
            # test_name
            "018. 3 infs, two infs can match with one ignored gt",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(2, 2), bottom_right=(11, 11)),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(2, 2), bottom_right=(11, 11)),
                ScoredLabeledBoundingBox(score=0.7, label="cow", top_left=(10, 10), bottom_right=(11, 11)),
            ],
            # ignored_ground_truths
            [
                LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                LabeledBoundingBox(label="cow", top_left=(2, 2), bottom_right=(11, 11)),
            ],
            # expected_matched
            [
                (
                    LabeledBoundingBox(label="cow", top_left=(2, 2), bottom_right=(11, 11)),
                    ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(2, 2), bottom_right=(11, 11)),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.7, label="cow", top_left=(10, 10), bottom_right=(11, 11))],
        ),
        (
            # test_name
            "018.1 3 infs, two infs can match with one ignored gt, order swap",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                LabeledBoundingBox(label="cow", top_left=(11, 11), bottom_right=(12, 12)),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                ScoredLabeledBoundingBox(score=0.8, label="cow", top_left=(2, 2), bottom_right=(11, 11)),
                ScoredLabeledBoundingBox(score=0.7, label="cow", top_left=(2, 3), bottom_right=(11, 11)),
            ],
            # ignored_ground_truths
            [
                LabeledBoundingBox(label="cow", top_left=(2, 2), bottom_right=(11, 11)),
            ],
            # expected_matched
            [
                (
                    LabeledBoundingBox(label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(1, 1), bottom_right=(11, 11)),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(label="cow", top_left=(11, 11), bottom_right=(12, 12)), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M01. Multi class no matches",
            # ground_truth
            [LabeledBoundingBox(top_left=(10, 10), bottom_right=(60, 60), label="cow")],
            # inference
            [ScoredLabeledBoundingBox(score=0.99, label="cow", top_left=(1, 1), bottom_right=(6, 6))],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(10, 10), bottom_right=(60, 60), label="cow"), None)],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.99, label="cow", top_left=(1, 1), bottom_right=(6, 6))],
        ),
        (
            # test_name
            "M01.1 Multi class no matches extra",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(10, 10), bottom_right=(60, 60), label="cow"),
                LabeledBoundingBox(top_left=(10, 10), bottom_right=(60, 60), label="dog"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.99, label="cow", top_left=(1, 1), bottom_right=(6, 6)),
                ScoredLabeledBoundingBox(score=0.98, label="fish", top_left=(5, 5), bottom_right=(6, 6)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (LabeledBoundingBox(top_left=(10, 10), bottom_right=(60, 60), label="cow"), None),
                (LabeledBoundingBox(top_left=(10, 10), bottom_right=(60, 60), label="dog"), None),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(score=0.99, label="cow", top_left=(1, 1), bottom_right=(6, 6)),
                ScoredLabeledBoundingBox(score=0.98, label="fish", top_left=(5, 5), bottom_right=(6, 6)),
            ],
        ),
        (
            # test_name
            "MP01.1 Multi class no matches extra",
            # ground_truth
            [
                LabeledPolygon(points=[(10, 10), (10, 60), (60, 60), (60, 10), (9, 9)], label="cow"),
                LabeledPolygon(points=[(10, 10), (10, 60), (60, 60), (60, 10), (10, 9)], label="dog"),
            ],
            # inference
            [
                ScoredLabeledPolygon(score=0.99, label="cow", points=[(1, 1), (1, 6), (6, 6), (6, 1), (6, 0)]),
                ScoredLabeledPolygon(score=0.98, label="fish", points=[(1, 1), (1, 6), (6, 6), (6, 1), (0, 0)]),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (LabeledPolygon(points=[(10, 10), (10, 60), (60, 60), (60, 10), (9, 9)], label="cow"), None),
                (LabeledPolygon(points=[(10, 10), (10, 60), (60, 60), (60, 10), (10, 9)], label="dog"), None),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledPolygon(score=0.99, label="cow", points=[(1, 1), (1, 6), (6, 6), (6, 1), (6, 0)]),
                ScoredLabeledPolygon(score=0.98, label="fish", points=[(1, 1), (1, 6), (6, 6), (6, 1), (0, 0)]),
            ],
        ),
        (
            # test_name
            "M02. Confused cows are confused",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(105, 105)),
                LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(113, 113)),
                LabeledBoundingBox(label="cow", top_left=(10, 10), bottom_right=(11, 11)),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(105, 105)),
                ScoredLabeledBoundingBox(score=0.7, label="cat", top_left=(99, 99), bottom_right=(105, 105)),
                ScoredLabeledBoundingBox(score=0.8, label="pig", top_left=(100, 100), bottom_right=(111, 111)),
                ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(100, 100), bottom_right=(113, 113)),
                ScoredLabeledBoundingBox(score=0.6, label="cow", top_left=(100, 100), bottom_right=(111, 111)),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(label="pig", top_left=(100, 100), bottom_right=(111, 111))],
            # expected_matched
            [
                (
                    LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(113, 113)),
                    ScoredLabeledBoundingBox(score=0.9, label="cow", top_left=(100, 100), bottom_right=(113, 113)),
                ),
                (
                    LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(105, 105)),
                    ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(105, 105)),
                ),
            ],
            # expected_unmatched_gt
            [
                (LabeledBoundingBox(label="cow", top_left=(10, 10), bottom_right=(11, 11)), None),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(score=0.7, label="cat", top_left=(99, 99), bottom_right=(105, 105)),
                ScoredLabeledBoundingBox(score=0.6, label="cow", top_left=(100, 100), bottom_right=(111, 111)),
            ],
        ),
        (
            # test_name
            "M21. empty gt",
            # ground_truth
            [],
            # inference
            [ScoredLabeledBoundingBox(score=0.99, label="cow", top_left=(1, 1), bottom_right=(6, 6))],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(score=0.99, label="cow", top_left=(1, 1), bottom_right=(6, 6))],
        ),
        (
            # test_name
            "M21.1 empty gt extra",
            # ground_truth
            [],
            # inference
            [
                ScoredLabeledBoundingBox(score=0.99, label="cow", top_left=(1, 1), bottom_right=(6, 6)),
                ScoredLabeledBoundingBox(score=0.98, label="dog", top_left=(1, 1), bottom_right=(6, 6)),
            ],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(score=0.99, label="cow", top_left=(1, 1), bottom_right=(6, 6)),
                ScoredLabeledBoundingBox(score=0.98, label="dog", top_left=(1, 1), bottom_right=(6, 6)),
            ],
        ),
        (
            # test_name
            "M22. empty inf",
            # ground_truth
            [LabeledBoundingBox(label="cow", top_left=(10, 10), bottom_right=(11, 11))],
            # inference
            [],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [(LabeledBoundingBox(label="cow", top_left=(10, 10), bottom_right=(11, 11)), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M22.1 empty inf extra",
            # ground_truth
            [
                LabeledBoundingBox(label="cow", top_left=(10, 10), bottom_right=(11, 11)),
                LabeledBoundingBox(label="dog", top_left=(10, 10), bottom_right=(11, 11)),
            ],
            # inference
            [],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (LabeledBoundingBox(label="cow", top_left=(10, 10), bottom_right=(11, 11)), None),
                (LabeledBoundingBox(label="dog", top_left=(10, 10), bottom_right=(11, 11)), None),
            ],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "MP22.1 empty inf extra",
            # ground_truth
            [
                LabeledPolygon(label="cow", points=[(10, 10), (11, 11), (11, 10)]),
                LabeledPolygon(label="dog", points=[(10, 10), (11, 11), (11, 10), (10, 8), (9, 8), (8, 8)]),
            ],
            # inference
            [],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (LabeledPolygon(label="cow", points=[(10, 10), (11, 11), (11, 10)]), None),
                (LabeledPolygon(label="dog", points=[(10, 10), (11, 11), (11, 10), (10, 8), (9, 8), (8, 8)]), None),
            ],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M23. all empty",
            # ground_truth
            [],
            # inference
            [],
            # ignored_ground_truths
            None,
            # expected_matched
            [],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "MP24. empty gt, but an ignored gt and an inf match",
            # ground_truth
            [],
            # inference
            [ScoredLabeledPolygon(points=[(10, 10), (11, 11), (11, 10)], label="x", score=0.9)],
            # ignored_ground_truths
            [
                LabeledPolygon(label="x", points=[(10, 10), (11, 11), (11, 10)]),
            ],
            # expected_matched
            [],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M24. empty gt, but an ignored gt and an inf match",
            # ground_truth
            [],
            # inference
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="x", score=0.9)],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="x")],
            # expected_matched
            [],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M25. empty gt, but an ignored gt and an inf match, diff label",
            # ground_truth
            [],
            # inference
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="x", score=0.9)],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="y")],
            # expected_matched
            [],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="x", score=0.9)],
        ),
        (
            # test_name
            "M31. 1 cow gt, 1 cat inf, iou < T",
            # ground_truth
            [LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow")],
            # inference
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9)],
            # ignored_ground_truths
            [],
            # expected_matched
            [],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"), None)],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9)],
        ),
        (
            # test_name
            "M32. 1 cow gt, 1 cat inf, iou > T, 1 confused match",
            # ground_truth
            [LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow")],
            # inference
            [ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9)],
            # ignored_ground_truths
            [],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9),
                ),
            ],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9)],
        ),
        (
            # test_name
            "M33. 2 cow gt, 2 cat inf, 2 iou < T",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(2, 2), bottom_right=(3, 3), label="cat", score=0.9),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"), None),
                (LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"), None),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(2, 2), bottom_right=(3, 3), label="cat", score=0.9),
            ],
        ),
        (
            # test_name
            "M34. 2 cow gt, 2 cat inf, one iou > T, one iou < T, 1 confused match",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9),
                ),
                (LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"), None),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9),
            ],
        ),
        (
            # test_name
            "MP34. 2 cow gt, 2 cat inf, one iou > T, one iou < T, 1 confused match",
            # ground_truth
            [
                LabeledPolygon(points=[(3, 3), (5, 3), (5, 5), (3, 5)], label="cow"),
                LabeledPolygon(points=[(13, 13), (15, 13), (15, 15), (13, 15)], label="cow"),
            ],
            # inference
            [
                ScoredLabeledPolygon(points=[(1, 1), (2, 1), (2, 2), (1, 2)], label="cat", score=0.9),
                ScoredLabeledPolygon(points=[(3, 3), (5, 3), (5, 5), (3, 5)], label="cat", score=0.9),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledPolygon(points=[(3, 3), (5, 3), (5, 5), (3, 5)], label="cow"),
                    ScoredLabeledPolygon(points=[(3, 3), (5, 3), (5, 5), (3, 5)], label="cat", score=0.9),
                ),
                (LabeledPolygon(points=[(13, 13), (15, 13), (15, 15), (13, 15)], label="cow"), None),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledPolygon(points=[(1, 1), (2, 1), (2, 2), (1, 2)], label="cat", score=0.9),
                ScoredLabeledPolygon(points=[(3, 3), (5, 3), (5, 5), (3, 5)], label="cat", score=0.9),
            ],
        ),
        (
            # test_name
            "M35. 2 cow gt, 2 cat inf, 2 iou > T, 2 confused matches",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.91),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.91),
                ),
                (
                    LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cat", score=0.9),
                ),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.91),
                ScoredLabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cat", score=0.9),
            ],
        ),
        (
            # test_name
            "M36. 1 confused match from one inf and many >T GT (highest IOU), all diff classes",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="rat"),
                LabeledBoundingBox(top_left=(2, 2), bottom_right=(5, 5), label="cow"),
                LabeledBoundingBox(top_left=(4, 4), bottom_right=(5, 5), label="dog"),
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(15, 15), label="fox"),
            ],
            # inference
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(5, 5), label="cat", score=0.9)],
            # ignored_ground_truths
            [],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledBoundingBox(top_left=(2, 2), bottom_right=(5, 5), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(5, 5), label="cat", score=0.9),
                ),
                (LabeledBoundingBox(top_left=(4, 4), bottom_right=(5, 5), label="dog"), None),
                (LabeledBoundingBox(top_left=(1, 1), bottom_right=(15, 15), label="fox"), None),
                (LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="rat"), None),
            ],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(5, 5), label="cat", score=0.9)],
        ),
        (
            # test_name
            "M37. 1 confused match from one gt and many >T inf (highest confidence), all diff classes",
            # ground_truth
            [LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow")],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog", score=0.2),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="fox", score=0.8),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="rat", score=0.7),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog", score=0.2),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="fox", score=0.8),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="rat", score=0.7),
            ],
        ),
        (
            # test_name
            "M41. One perfect match, one unmatched gt of a different class",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog"),
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog"), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M42. One perfect match, one unmatched inf of a different class",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog", score=0.9),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog", score=0.9)],
        ),
        (
            # test_name
            "M43. One perfect match, one unmatched gt of the same class bc lower iou",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(11, 11), bottom_right=(21, 21), label="cow"),
                LabeledBoundingBox(top_left=(11, 11), bottom_right=(20, 20), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(11, 11), bottom_right=(21, 21), label="cow", score=0.9),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(11, 11), bottom_right=(21, 21), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(11, 11), bottom_right=(21, 21), label="cow", score=0.9),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(11, 11), bottom_right=(20, 20), label="cow"), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M44. One perfect match, one unmatched inf of the same class bc lower confidence despite higher iou",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="cow", score=0.5),
                ScoredLabeledBoundingBox(top_left=(10, 10), bottom_right=(20, 20), label="cow", score=0.9),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(10, 10), bottom_right=(20, 20), label="cow", score=0.9),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="cow", score=0.5)],
        ),
        (
            # test_name
            "MP44. One perfect match, one unmatched inf of the same class bc lower confidence despite higher iou",
            # ground_truth
            [
                LabeledPolygon(points=[(10, 10), (10, 22), (22, 22), (22, 10)], label="cow"),
            ],
            # inference
            [
                ScoredLabeledPolygon(points=[(10, 10), (10, 22), (22, 22), (22, 10)], label="cow", score=0.5),
                ScoredLabeledPolygon(points=[(10, 10), (10, 20), (20, 20), (20, 10)], label="cow", score=0.9),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [
                (
                    LabeledPolygon(points=[(10, 10), (10, 22), (22, 22), (22, 10)], label="cow"),
                    ScoredLabeledPolygon(points=[(10, 10), (10, 20), (20, 20), (20, 10)], label="cow", score=0.9),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledPolygon(points=[(10, 10), (10, 22), (22, 22), (22, 10)], label="cow", score=0.5)],
        ),
        (
            # test_name
            "M531. 1 cow gt, cat cow inf, iou < T",
            # ground_truth
            [LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow")],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow", score=0.6),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"), None)],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9)],
        ),
        (
            # test_name
            "M532. 1 cow gt, cat cow inf, iou > T, 1 confused match",
            # ground_truth
            [LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow")],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow", score=0.6),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9),
                ),
            ],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9)],
        ),
        (
            # test_name
            "M533. 2 cow gt, 2 cat 1 cow inf, 2 iou < T",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow", score=0.6),
                ScoredLabeledBoundingBox(top_left=(2, 2), bottom_right=(3, 3), label="cat", score=0.9),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"), None),
                (LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"), None),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(2, 2), bottom_right=(3, 3), label="cat", score=0.9),
            ],
        ),
        (
            # test_name
            "M534. 2 cow gt, 2 cat 1 cow inf, one iou > T, one iou < T, 1 confused match",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow", score=0.6),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9),
                ),
                (LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"), None),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.9),
            ],
        ),
        (
            # test_name
            "MP534. 2 cow gt, 2 cat 1 cow inf, one iou > T, one iou < T, 1 confused match",
            # ground_truth
            [
                LabeledPolygon(points=[(3, 3), (5, 3), (5, 5), (3, 5)], label="cow"),
                LabeledPolygon(points=[(13, 13), (15, 13), (15, 15), (13, 15)], label="cow"),
            ],
            # inference
            [
                ScoredLabeledPolygon(points=[(1, 1), (2, 1), (2, 2), (1, 2)], label="cat", score=0.9),
                ScoredLabeledPolygon(points=[(3, 3), (5, 3), (5, 5), (3, 5)], label="cat", score=0.9),
                ScoredLabeledPolygon(points=[(55, 55), (77, 55), (77, 77), (55, 77)], label="cow", score=0.6),
            ],
            # ignored_ground_truths
            [LabeledPolygon(points=[(55, 55), (77, 55), (77, 77), (55, 77)], label="cow")],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledPolygon(points=[(3, 3), (5, 3), (5, 5), (3, 5)], label="cow"),
                    ScoredLabeledPolygon(points=[(3, 3), (5, 3), (5, 5), (3, 5)], label="cat", score=0.9),
                ),
                (LabeledPolygon(points=[(13, 13), (15, 13), (15, 15), (13, 15)], label="cow"), None),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledPolygon(points=[(1, 1), (2, 1), (2, 2), (1, 2)], label="cat", score=0.9),
                ScoredLabeledPolygon(points=[(3, 3), (5, 3), (5, 5), (3, 5)], label="cat", score=0.9),
            ],
        ),
        (
            # test_name
            "M535. 2 cow gt, 2 cat 1 cow inf, 2 iou > T, 2 confused matches",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow", score=0.6),
                ScoredLabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.91),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.91),
                ),
                (
                    LabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cat", score=0.9),
                ),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="cat", score=0.91),
                ScoredLabeledBoundingBox(top_left=(13, 13), bottom_right=(15, 15), label="cat", score=0.9),
            ],
        ),
        (
            # test_name
            "M536. 1 confused match from one inf and many >T GT (highest IOU), all diff classes",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="rat"),
                LabeledBoundingBox(top_left=(2, 2), bottom_right=(5, 5), label="cow"),
                LabeledBoundingBox(top_left=(4, 4), bottom_right=(5, 5), label="dog"),
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(15, 15), label="fox"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(5, 5), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow", score=0.6),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledBoundingBox(top_left=(2, 2), bottom_right=(5, 5), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(5, 5), label="cat", score=0.9),
                ),
                (LabeledBoundingBox(top_left=(4, 4), bottom_right=(5, 5), label="dog"), None),
                (LabeledBoundingBox(top_left=(1, 1), bottom_right=(15, 15), label="fox"), None),
                (LabeledBoundingBox(top_left=(3, 3), bottom_right=(5, 5), label="rat"), None),
            ],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(5, 5), label="cat", score=0.9)],
        ),
        (
            # test_name
            "M537. 1 confused match from one gt and many >T inf (highest confidence), all diff classes",
            # ground_truth
            [LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow")],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog", score=0.2),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="fox", score=0.8),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow", score=0.6),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="rat", score=0.7),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (
                    LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cat", score=0.9),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog", score=0.2),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="fox", score=0.8),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="rat", score=0.7),
            ],
        ),
        (
            # test_name
            "M541. One perfect match, one unmatched gt of a different class",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog"),
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9),
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow", score=0.6),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog"), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M542. One perfect match, one unmatched inf of a different class",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow", score=0.6),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9),
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog", score=0.9),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog", score=0.9)],
        ),
        (
            # test_name
            "M543. One perfect match, one unmatched gt of the same class bc lower iou",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(11, 11), bottom_right=(21, 21), label="cow"),
                LabeledBoundingBox(top_left=(11, 11), bottom_right=(20, 20), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(11, 11), bottom_right=(21, 21), label="cow", score=0.9),
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow", score=0.6),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(11, 11), bottom_right=(21, 21), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(11, 11), bottom_right=(21, 21), label="cow", score=0.9),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(11, 11), bottom_right=(20, 20), label="cow"), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M544. One perfect match, one unmatched inf of the same class bc lower confidence despite higher iou",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="cow", score=0.5),
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow", score=0.6),
                ScoredLabeledBoundingBox(top_left=(10, 10), bottom_right=(20, 20), label="cow", score=0.9),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(10, 10), bottom_right=(20, 20), label="cow", score=0.9),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="cow", score=0.5)],
        ),
        (
            # test_name
            "MP544. One perfect match, one unmatched inf of the same class bc lower confidence despite higher iou",
            # ground_truth
            [
                LabeledPolygon(points=[(10, 10), (10, 22), (22, 22), (22, 10)], label="cow"),
            ],
            # inference
            [
                ScoredLabeledPolygon(points=[(10, 10), (10, 22), (22, 22), (22, 10)], label="cow", score=0.5),
                ScoredLabeledPolygon(points=[(55, 55), (77, 55), (77, 77), (55, 77)], label="cow", score=0.6),
                ScoredLabeledPolygon(points=[(10, 10), (10, 20), (20, 20), (20, 10)], label="cow", score=0.9),
            ],
            # ignored_ground_truths
            [
                LabeledPolygon(points=[(55, 55), (77, 55), (77, 77), (55, 77)], label="cow"),
            ],
            # expected_matched
            [
                (
                    LabeledPolygon(points=[(10, 10), (10, 22), (22, 22), (22, 10)], label="cow"),
                    ScoredLabeledPolygon(points=[(10, 10), (10, 20), (20, 20), (20, 10)], label="cow", score=0.9),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [ScoredLabeledPolygon(points=[(10, 10), (10, 22), (22, 22), (22, 10)], label="cow", score=0.5)],
        ),
        (
            # test_name
            "M52. One ignored gt sorta matching with an inf, but not stealing from another gt",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(55, 55), bottom_right=(75, 75), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(75, 75), label="cow", score=0.61),
            ],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(77, 77), label="cow")],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(55, 55), bottom_right=(75, 75), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(75, 75), label="cow", score=0.61),
                ),
            ],
            # expected_unmatched_gt
            [],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M53. One ignored gt sorta matching with an inf, but stealing from another gt",
            # ground_truth
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(76, 76), label="cow")],
            # inference
            [ScoredLabeledBoundingBox(top_left=(55, 55), bottom_right=(75, 75), label="cow", score=0.61)],
            # ignored_ground_truths
            [LabeledBoundingBox(top_left=(55, 55), bottom_right=(75, 75), label="cow")],
            # expected_matched
            [],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(55, 55), bottom_right=(76, 76), label="cow"), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M61. large flower petal arrangement clean matches",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(55, 55), bottom_right=(70, 70), label="cow"),
                LabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 250), label="hat"),
                LabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 300), label="man"),
                LabeledBoundingBox(top_left=(200, 280), bottom_right=(280, 330), label="pants"),
                LabeledBoundingBox(top_left=(100, 100), bottom_right=(200, 200), label="man"),
                LabeledBoundingBox(top_left=(120, 100), bottom_right=(200, 220), label="man"),
                LabeledBoundingBox(top_left=(400, 400), bottom_right=(450, 450), label="flower"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(399, 401), bottom_right=(449, 448), label="flower", score=0.99),
                ScoredLabeledBoundingBox(top_left=(399, 401), bottom_right=(449, 448), label="petal", score=0.98),
                ScoredLabeledBoundingBox(top_left=(399, 401), bottom_right=(449, 448), label="petal", score=0.975),
                ScoredLabeledBoundingBox(top_left=(400, 400), bottom_right=(450, 450), label="petal", score=0.97),
                ScoredLabeledBoundingBox(top_left=(350, 350), bottom_right=(450, 450), label="petal", score=0.96),
                ScoredLabeledBoundingBox(top_left=(300, 300), bottom_right=(500, 500), label="petal", score=0.95),
                ScoredLabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 290), label="hat", score=0.8),
                ScoredLabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 300), label="man", score=0.79),
                ScoredLabeledBoundingBox(top_left=(100, 100), bottom_right=(200, 200), label="man", score=0.78),
                ScoredLabeledBoundingBox(top_left=(1, 2), bottom_right=(3, 4), label="man", score=0.77),
                ScoredLabeledBoundingBox(top_left=(1, 12), bottom_right=(3, 14), label="man", score=0.76),
                ScoredLabeledBoundingBox(top_left=(11, 2), bottom_right=(13, 4), label="man", score=0.75),
                ScoredLabeledBoundingBox(top_left=(54, 54), bottom_right=(72, 72), label="cow", score=0.6),
            ],
            # ignored_ground_truths
            [
                LabeledBoundingBox(top_left=(402, 403), bottom_right=(444, 450), label="petal"),
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(3, 4), label="man"),
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(1000, 1000), label="random"),
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(1000, 1000), label="man"),
            ],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(55, 55), bottom_right=(70, 70), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(54, 54), bottom_right=(72, 72), label="cow", score=0.6),
                ),
                (
                    LabeledBoundingBox(top_left=(400, 400), bottom_right=(450, 450), label="flower"),
                    ScoredLabeledBoundingBox(top_left=(399, 401), bottom_right=(449, 448), label="flower", score=0.99),
                ),
                (
                    LabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 250), label="hat"),
                    ScoredLabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 290), label="hat", score=0.8),
                ),
                (
                    LabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 300), label="man"),
                    ScoredLabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 300), label="man", score=0.79),
                ),
                (
                    LabeledBoundingBox(top_left=(100, 100), bottom_right=(200, 200), label="man"),
                    ScoredLabeledBoundingBox(top_left=(100, 100), bottom_right=(200, 200), label="man", score=0.78),
                ),
            ],
            # expected_unmatched_gt
            [
                (LabeledBoundingBox(top_left=(120, 100), bottom_right=(200, 220), label="man"), None),
                (LabeledBoundingBox(top_left=(200, 280), bottom_right=(280, 330), label="pants"), None),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(top_left=(1, 12), bottom_right=(3, 14), label="man", score=0.76),
                ScoredLabeledBoundingBox(top_left=(11, 2), bottom_right=(13, 4), label="man", score=0.75),
                ScoredLabeledBoundingBox(top_left=(350, 350), bottom_right=(450, 450), label="petal", score=0.96),
                ScoredLabeledBoundingBox(top_left=(300, 300), bottom_right=(500, 500), label="petal", score=0.95),
            ],
        ),
        (
            # test_name
            "M62. ignore man",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(55, 55), bottom_right=(70, 70), label="cow"),
                LabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 280), label="hat"),
                LabeledBoundingBox(top_left=(200, 280), bottom_right=(280, 330), label="pants"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(399, 401), bottom_right=(449, 448), label="flower", score=0.99),
                ScoredLabeledBoundingBox(top_left=(399, 401), bottom_right=(449, 448), label="petal", score=0.98),
                ScoredLabeledBoundingBox(top_left=(399, 401), bottom_right=(449, 448), label="petal", score=0.975),
                ScoredLabeledBoundingBox(top_left=(400, 400), bottom_right=(450, 450), label="petal", score=0.97),
                ScoredLabeledBoundingBox(top_left=(350, 350), bottom_right=(450, 450), label="petal", score=0.96),
                ScoredLabeledBoundingBox(top_left=(300, 300), bottom_right=(500, 500), label="petal", score=0.95),
                ScoredLabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 290), label="hat", score=0.8),
                ScoredLabeledBoundingBox(top_left=(54, 54), bottom_right=(72, 72), label="cow", score=0.6),
                ScoredLabeledBoundingBox(top_left=(199, 280), bottom_right=(280, 331), label="dress", score=0.5),
            ],
            # ignored_ground_truths
            [
                LabeledBoundingBox(top_left=(402, 403), bottom_right=(444, 450), label="petal"),
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(1000, 1000), label="random"),
                LabeledBoundingBox(top_left=(400, 400), bottom_right=(450, 450), label="flower"),
            ],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(55, 55), bottom_right=(70, 70), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(54, 54), bottom_right=(72, 72), label="cow", score=0.6),
                ),
                (
                    LabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 280), label="hat"),
                    ScoredLabeledBoundingBox(top_left=(200, 200), bottom_right=(280, 290), label="hat", score=0.8),
                ),
            ],
            # expected_unmatched_gt
            [
                (
                    LabeledBoundingBox(top_left=(200, 280), bottom_right=(280, 330), label="pants"),
                    ScoredLabeledBoundingBox(top_left=(199, 280), bottom_right=(280, 331), label="dress", score=0.5),
                ),
            ],
            # expected_unmatched_inf
            [
                ScoredLabeledBoundingBox(top_left=(199, 280), bottom_right=(280, 331), label="dress", score=0.5),
                ScoredLabeledBoundingBox(top_left=(350, 350), bottom_right=(450, 450), label="petal", score=0.96),
                ScoredLabeledBoundingBox(top_left=(300, 300), bottom_right=(500, 500), label="petal", score=0.95),
            ],
        ),
        (
            # test_name
            "M71. Valid match of the same class but no match bc lower confidence / iou",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(100, 100), label="A"),
                LabeledBoundingBox(top_left=(15, 1), bottom_right=(115, 100), label="A"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(5, 1), bottom_right=(105, 100), label="A", score=0.99),
                ScoredLabeledBoundingBox(top_left=(5, 5), bottom_right=(105, 105), label="A", score=0.98),
            ],
            # ignored_ground_truths
            [],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(1, 1), bottom_right=(100, 100), label="A"),
                    ScoredLabeledBoundingBox(top_left=(5, 1), bottom_right=(105, 100), label="A", score=0.99),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(15.0, 1.0), bottom_right=(115.0, 100.0), label="A"), None)],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(5, 5), bottom_right=(105, 105), label="A", score=0.98)],
        ),
    ],
)
def test__match_inferences_multiclass(
    test_name: str,
    ground_truths: List[GT],
    inferences: List[Inf],
    ignored_ground_truths: Optional[List[GT]],
    expected_matched: List[Tuple[GT, Inf]],
    expected_unmatched_gt: List[GT],
    expected_unmatched_inf: List[Inf],
) -> None:
    matches = match_inferences_multiclass(
        ground_truths,
        inferences,
        ignored_ground_truths=ignored_ground_truths,
        mode="pascal",
        iou_threshold=0.5,
    )

    assert expected_matched == matches.matched
    assert expected_unmatched_gt == matches.unmatched_gt
    assert expected_unmatched_inf == matches.unmatched_inf


def test__match_inferences_multiclass__invalid_mode() -> None:
    with pytest.raises(InputValidationError):
        match_inferences_multiclass(
            [LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110))],
            [ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112))],
            mode="not pascal",
        )


@pytest.mark.parametrize(
    "test_name, ground_truths, inferences, expected_matched, expected_unmatched_gt, expected_unmatched_inf",
    [
        (
            # test_name
            "M41. One perfect match, one unmatched gt of a different class - iou max",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog"),
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
            ],
            # inference
            [
                ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9),
            ],
            # expected_matched
            [
                (
                    LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow"),
                    ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9),
                ),
            ],
            # expected_unmatched_gt
            [(LabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="dog"), None)],
            # expected_unmatched_inf
            [],
        ),
        (
            # test_name
            "M41.1 No match, one unmatched gt of a different class - iou max",
            # ground_truth
            [
                LabeledBoundingBox(top_left=(1, 1), bottom_right=(3, 3), label="dog"),
                LabeledBoundingBox(top_left=(0, 0), bottom_right=(2, 2), label="cow"),
            ],
            # inference
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9)],
            # expected_matched
            [],
            # expected_unmatched_gt
            [
                (LabeledBoundingBox(top_left=(0, 0), bottom_right=(2, 2), label="cow"), None),
                (LabeledBoundingBox(top_left=(1, 1), bottom_right=(3, 3), label="dog"), None),
            ],
            # expected_unmatched_inf
            [ScoredLabeledBoundingBox(top_left=(1, 1), bottom_right=(2, 2), label="cow", score=0.9)],
        ),
    ],
)
def test__match_inferences_multiclass__iou(
    test_name: str,
    ground_truths: List[Union[LabeledBoundingBox, LabeledPolygon]],
    inferences: List[Union[ScoredLabeledBoundingBox, ScoredLabeledPolygon]],
    expected_matched: List[Tuple[GT, Inf]],
    expected_unmatched_gt: List[GT],
    expected_unmatched_inf: List[Inf],
) -> None:
    matches = match_inferences_multiclass(
        ground_truths,
        inferences,
        mode="pascal",
        iou_threshold=0.999,
    )

    assert expected_matched == matches.matched
    assert expected_unmatched_gt == matches.unmatched_gt
    assert expected_unmatched_inf == matches.unmatched_inf
