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
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import pytest

from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import Polygon
from kolena.workflow.annotation import ScoredBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.metrics._geometry import GT
from kolena.workflow.metrics._geometry import Inf
from kolena.workflow.metrics._geometry import InferenceMatches
from kolena.workflow.metrics._geometry import iou
from kolena.workflow.metrics._geometry import match_inferences


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
def test_iou(points1: Union[BoundingBox, Polygon], points2: Union[BoundingBox, Polygon], expected_iou: float) -> None:
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
            "8. 2 inf, 2 matching GT, higher conf has lower IOU => higher conf inf matched",
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
            "9. 2 inf, 2 matching GT, higher conf has higher IOU => higher conf inf matched",
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
            "10.1 Both inferences match to a GT => higher confidence inference matched",
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
def test_match_inferences_single_class(
    test_name: str,
    ground_truths: List[GT],
    inferences: List[Inf],
    ignored_ground_truths: Optional[List[GT]],
    expected_matched: List[Tuple[GT, Inf]],
    expected_unmatched_gt: List[GT],
    expected_unmatched_inf: List[Inf],
) -> None:
    matches: InferenceMatches = match_inferences(
        ground_truths,
        inferences,
        ignored_ground_truths=ignored_ground_truths,
    )
    assert expected_matched == matches.matched
    assert expected_unmatched_gt == matches.unmatched_gt
    assert expected_unmatched_inf == matches.unmatched_inf


@pytest.mark.parametrize(
    "test_name, ground_truths, inferences",
    [
        (
            # test_name
            "15. Fail mode",
            # ground_truth
            [LabeledBoundingBox(label="cow", top_left=(100, 100), bottom_right=(110, 110))],
            # inference
            [ScoredLabeledBoundingBox(score=0.5, label="cow", top_left=(99, 99), bottom_right=(112, 112))],
        ),
    ],
)
def test_match_inferences_single_class_fails(
    test_name: str,
    ground_truths: List[GT],
    inferences: List[Inf],
) -> None:
    with pytest.raises(Exception):
        match_inferences(
            ground_truths,
            inferences,
            mode="not pascal",
        )
