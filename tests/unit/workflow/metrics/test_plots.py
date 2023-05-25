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

import pytest

from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledPolygon
from kolena.workflow.evaluator import ConfusionMatrix
from kolena.workflow.metrics._geometry import MulticlassInferenceMatches
from kolena.workflow.metrics._plots import compute_test_case_confusion_matrix


@pytest.mark.parametrize(
    "test_name, matchings, ordered_labels, matrix",
    [
        (
            "zeros",
            [
                MulticlassInferenceMatches(
                    matched=[],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), None),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [0, 0],
                [0, 0],
            ],
        ),
        (
            "zeros with unmatched inferences",
            [
                MulticlassInferenceMatches(
                    matched=[],
                    unmatched_gt=[],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0),
                        ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0),
                    ],
                ),
            ],
            ["a", "b"],
            [
                [0, 0],
                [0, 0],
            ],
        ),
        (
            "zeros with unmatched gt and unmatched inf",
            [
                MulticlassInferenceMatches(
                    matched=[],
                    unmatched_gt=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), None),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0),
                    ],
                ),
            ],
            ["a", "b"],
            [
                [0, 0],
                [0, 0],
            ],
        ),
        (
            "zeros with two matchings",
            [
                MulticlassInferenceMatches(
                    matched=[],
                    unmatched_gt=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), None),
                    ],
                    unmatched_inf=[],
                ),
                MulticlassInferenceMatches(
                    matched=[],
                    unmatched_gt=[],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0),
                    ],
                ),
            ],
            ["a", "b"],
            [
                [0, 0],
                [0, 0],
            ],
        ),
        (
            "zeros, but one match for label a",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((3, 3), (4, 4), "a"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), None),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [1, 0],
                [0, 0],
            ],
        ),
        (
            "zeros, but one match for label b",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), None),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [0, 0],
                [0, 1],
            ],
        ),
        (
            "zeros, but b is confused with a",
            [
                MulticlassInferenceMatches(
                    matched=[],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), None),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0)),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [0, 1],
                [0, 0],
            ],
        ),
        (
            "zeros, but a is confused with b",
            [
                MulticlassInferenceMatches(
                    matched=[],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0)),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [0, 0],
                [1, 0],
            ],
        ),
        (
            "no confusion, one TP per label",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0)),
                        (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), None),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [1, 0],
                [0, 1],
            ],
        ),
        (
            "only confusion",
            [
                MulticlassInferenceMatches(
                    matched=[],
                    unmatched_gt=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                        (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0)),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [0, 1],
                [1, 0],
            ],
        ),
        (
            "only confusion, one TP for a",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                        (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0)),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [1, 1],
                [1, 0],
            ],
        ),
        (
            "only confusion, one TP for b",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                        (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0)),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [0, 1],
                [1, 1],
            ],
        ),
        (
            "ones",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0)),
                        (LabeledBoundingBox((7, 7), (8, 8), "b"), ScoredLabeledBoundingBox((7, 7), (8, 8), "b", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                        (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0)),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [1, 1],
                [1, 1],
            ],
        ),
        (
            "ones, with two matchings, TPs",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0)),
                        (LabeledBoundingBox((7, 7), (8, 8), "b"), ScoredLabeledBoundingBox((7, 7), (8, 8), "b", 0)),
                    ],
                    unmatched_gt=[],
                    unmatched_inf=[],
                ),
                MulticlassInferenceMatches(
                    matched=[],
                    unmatched_gt=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                        (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0)),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [1, 1],
                [1, 1],
            ],
        ),
        (
            "ones, with two matchings, mixed",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0)),
                    ],
                    unmatched_inf=[],
                ),
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((7, 7), (8, 8), "b"), ScoredLabeledBoundingBox((7, 7), (8, 8), "b", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                    ],
                    unmatched_inf=[],
                ),
            ],
            ["a", "b"],
            [
                [1, 1],
                [1, 1],
            ],
        ),
        (
            "two single class matchings",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((3, 3), (4, 4), "a"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                        (LabeledBoundingBox((6, 6), (7, 7), "a"), ScoredLabeledBoundingBox((6, 6), (7, 7), "a", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "a", 0),
                    ],
                ),
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (4, 4), "b"), ScoredLabeledBoundingBox((1, 1), (4, 4), "b", 0)),
                        (LabeledBoundingBox((2, 2), (7, 7), "b"), ScoredLabeledBoundingBox((2, 2), (7, 7), "b", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), None),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "b", 0),
                    ],
                ),
            ],
            ["a", "b"],
            [
                [2, 0],
                [0, 2],
            ],
        ),
        (
            "large",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (
                            LabeledBoundingBox(top_left=(1.0, 1.0), bottom_right=(6.0, 6.0), label="cow"),
                            ScoredLabeledBoundingBox((1.0, 1.0), (6.0, 6.0), "cow", 0.9),
                        ),
                        (
                            LabeledBoundingBox(top_left=(10.0, 10.0), bottom_right=(22.0, 22.0), label="cow"),
                            ScoredLabeledBoundingBox((10.0, 10.0), (20.0, 20.0), "cow", 0.9),
                        ),
                    ],
                    unmatched_gt=[
                        (
                            LabeledPolygon(
                                points=[(1, 1), (1, 2), (2, 2), (2, 1)],
                                label="cow",
                            ),
                            ScoredLabeledPolygon(
                                points=[(1, 1), (1, 20), (20, 20), (20, 1)],
                                label="dog",
                                score=0.9,
                            ),
                        ),
                    ],
                    unmatched_inf=[],
                ),
                MulticlassInferenceMatches(
                    matched=[
                        (
                            LabeledBoundingBox(top_left=(10.0, 10.0), bottom_right=(22.0, 22.0), label="cow"),
                            ScoredLabeledBoundingBox((10.0, 10.0), (20.0, 20.0), "cow", 0.9),
                        ),
                    ],
                    unmatched_gt=[
                        (
                            LabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="fish"),
                            ScoredLabeledBoundingBox((10, 10), (20, 20), "dog", 0.9),
                        ),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((10.0, 10.0), (22.0, 22.0), "cat", 0.5),
                    ],
                ),
                MulticlassInferenceMatches(
                    matched=[
                        (
                            LabeledPolygon(
                                points=[(10.0, 10.0), (10.0, 22.0), (22.0, 22.0), (22.0, 10.0)],
                                label="cow",
                            ),
                            ScoredLabeledPolygon(
                                points=[(10.0, 10.0), (10.0, 20.0), (20.0, 20.0), (20.0, 10.0)],
                                label="cow",
                                score=0.9,
                            ),
                        ),
                    ],
                    unmatched_gt=[],
                    unmatched_inf=[
                        ScoredLabeledPolygon(
                            points=[(10.0, 10.0), (10.0, 22.0), (22.0, 22.0), (22.0, 10.0)],
                            label="cow",
                            score=0.5,
                        ),
                    ],
                ),
                MulticlassInferenceMatches(
                    matched=[
                        (
                            LabeledPolygon(
                                points=[(10.0, 10.0), (10.0, 22.0), (22.0, 22.0), (22.0, 10.0)],
                                label="cat",
                            ),
                            ScoredLabeledPolygon(
                                points=[(10.0, 10.0), (10.0, 20.0), (20.0, 20.0), (20.0, 10.0)],
                                label="cat",
                                score=0.9,
                            ),
                        ),
                    ],
                    unmatched_gt=[],
                    unmatched_inf=[
                        ScoredLabeledPolygon(
                            points=[(10.0, 10.0), (10.0, 22.0), (22.0, 22.0), (22.0, 10.0)],
                            label="dog",
                            score=0.5,
                        ),
                    ],
                ),
                MulticlassInferenceMatches(
                    matched=[],
                    unmatched_gt=[
                        (
                            LabeledPolygon(
                                points=[(10, 10), (10, 22), (22, 22), (22, 10)],
                                label="cow",
                            ),
                            ScoredLabeledPolygon(
                                points=[(10, 10), (10, 20), (20, 20), (20, 10)],
                                label="cat",
                                score=0.9,
                            ),
                        ),
                        (
                            LabeledPolygon(
                                points=[(1, 1), (1, 2), (2, 2), (2, 1)],
                                label="cow",
                            ),
                            ScoredLabeledPolygon(
                                points=[(1, 1), (1, 2), (2, 2), (2, 1)],
                                label="dog",
                                score=0.9,
                            ),
                        ),
                    ],
                    unmatched_inf=[
                        ScoredLabeledPolygon(
                            points=[(1, 1), (1, 2), (2, 2), (2, 1)],
                            label="dog",
                            score=0.5,
                        ),
                    ],
                ),
            ],
            ["cat", "cow", "dog", "fish"],
            [
                [1, 0, 0, 0],
                [1, 4, 2, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
            ],
        ),
    ],
)
def test__confusion__matrix(
    test_name: str,
    matchings: List[MulticlassInferenceMatches],
    ordered_labels: List[str],
    matrix: List[List[int]],
) -> None:
    conf_mat = compute_test_case_confusion_matrix(all_matches=matchings, test_case_name=test_name, plot_title=test_name)
    assert conf_mat == ConfusionMatrix(title=test_name, labels=ordered_labels, matrix=matrix)


@pytest.mark.parametrize(
    "test_name, matchings",
    [
        (
            "one class invalid",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((3, 3), (4, 4), "a"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                        (LabeledBoundingBox((6, 6), (7, 7), "a"), ScoredLabeledBoundingBox((6, 6), (7, 7), "a", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "a", 0),
                    ],
                ),
            ],
        ),
        (
            "12 classes invalid",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((3, 3), (4, 4), "a"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                        (LabeledBoundingBox((6, 6), (7, 7), "a"), ScoredLabeledBoundingBox((6, 6), (7, 7), "a", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "2", 0),
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "3", 0),
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "4", 0),
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "5", 0),
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "6", 0),
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "7", 0),
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "8", 0),
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "9", 0),
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "10", 0),
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "11", 0),
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "12", 0),
                    ],
                ),
            ],
        ),
    ],
)
def test__confusion__matrix__fails(
    test_name: str,
    matchings: MulticlassInferenceMatches,
) -> None:
    conf_mat = compute_test_case_confusion_matrix(all_matches=matchings, test_case_name=test_name, plot_title=test_name)
    assert conf_mat is None
