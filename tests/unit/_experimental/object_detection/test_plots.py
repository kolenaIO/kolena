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
from typing import Dict
from typing import List
from typing import Union

import pytest

from kolena.workflow import ConfusionMatrix
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledPolygon
from kolena.workflow.metrics import InferenceMatches
from kolena.workflow.metrics import MulticlassInferenceMatches

TOLERANCE = 1e-12


def assert_curves(
    curves: List[Curve],
    expected: List[Curve],
) -> None:
    assert len(curves) == len(expected)
    for curve, expectation in zip(curves, expected):
        assert curve.label == expectation.label
        assert len(curve.x) == len(expectation.x)
        assert sum(abs(a - b) for a, b in zip(curve.x, expectation.x)) < TOLERANCE
        assert len(curve.y) == len(expectation.y)
        assert sum(abs(a - b) for a, b in zip(curve.y, expectation.y)) < TOLERANCE
        for extra_key in curve.extra.keys():
            assert sum(abs(a - b) for a, b in zip(curve.extra[extra_key], expectation.extra[extra_key])) < TOLERANCE


def assert_curve_plot_equals_expected(
    plot: CurvePlot,
    expected: CurvePlot,
) -> None:
    assert plot.title == expected.title
    assert plot.x_label == expected.x_label
    assert plot.y_label == expected.y_label
    assert_curves(plot.curves, expected.curves)
    assert plot.x_config == expected.x_config
    assert plot.y_config == expected.y_config


TEST_MATCHING: Dict[str, List[Union[MulticlassInferenceMatches, InferenceMatches]]] = {
    "zeros with unmatched gt and unmatched inf": [
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
    "zeros with two matchings": [
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
    "zeros, but one match for label a": [
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
    "zeros, but one match for label b": [
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
    "zeros, but b is confused with a": [
        MulticlassInferenceMatches(
            matched=[],
            unmatched_gt=[
                (LabeledBoundingBox((1, 1), (2, 2), "b"), None),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1)),
            ],
            unmatched_inf=[ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1)],
        ),
    ],
    "zeros, but a is confused with b": [
        MulticlassInferenceMatches(
            matched=[],
            unmatched_gt=[
                (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.3)),
            ],
            unmatched_inf=[ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.3)],
        ),
    ],
    "no confusion, one TP per label": [
        MulticlassInferenceMatches(
            matched=[
                (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.5)),
                (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.6)),
            ],
            unmatched_gt=[
                (LabeledBoundingBox((1, 1), (2, 2), "b"), None),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
            ],
            unmatched_inf=[],
        ),
    ],
    "only confusion": [
        MulticlassInferenceMatches(
            matched=[],
            unmatched_gt=[
                (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.7)),
                (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.8)),
            ],
            unmatched_inf=[
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.7),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.8),
            ],
        ),
    ],
    "only confusion, one TP for a": [
        MulticlassInferenceMatches(
            matched=[
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9)),
            ],
            unmatched_gt=[
                (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.8)),
                (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.7)),
            ],
            unmatched_inf=[
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.8),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.7),
            ],
        ),
    ],
    "only confusion, one TP for b": [
        MulticlassInferenceMatches(
            matched=[
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.9)),
            ],
            unmatched_gt=[
                (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.8)),
                (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.7)),
            ],
            unmatched_inf=[
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.8),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.7),
            ],
        ),
    ],
    "ones": [
        MulticlassInferenceMatches(
            matched=[
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9)),
                (LabeledBoundingBox((7, 7), (8, 8), "b"), ScoredLabeledBoundingBox((7, 7), (8, 8), "b", 0.8)),
            ],
            unmatched_gt=[
                (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.7)),
                (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.6)),
            ],
            unmatched_inf=[
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.7),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.6),
            ],
        ),
    ],
    "ones, with two matchings, TPs": [
        MulticlassInferenceMatches(
            matched=[
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9)),
                (LabeledBoundingBox((7, 7), (8, 8), "b"), ScoredLabeledBoundingBox((7, 7), (8, 8), "b", 0.8)),
            ],
            unmatched_gt=[],
            unmatched_inf=[],
        ),
        MulticlassInferenceMatches(
            matched=[],
            unmatched_gt=[
                (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.7)),
                (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.6)),
            ],
            unmatched_inf=[
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.7),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.6),
            ],
        ),
    ],
    "ones, with two matchings, mixed": [
        MulticlassInferenceMatches(
            matched=[
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9)),
            ],
            unmatched_gt=[
                (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.8)),
            ],
            unmatched_inf=[ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.8)],
        ),
        MulticlassInferenceMatches(
            matched=[
                (LabeledBoundingBox((7, 7), (8, 8), "b"), ScoredLabeledBoundingBox((7, 7), (8, 8), "b", 0.6)),
            ],
            unmatched_gt=[
                (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.5)),
            ],
            unmatched_inf=[ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.5)],
        ),
    ],
    "two single class matchings": [
        MulticlassInferenceMatches(
            matched=[
                (LabeledBoundingBox((3, 3), (4, 4), "a"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.9)),
                (LabeledBoundingBox((6, 6), (7, 7), "a"), ScoredLabeledBoundingBox((6, 6), (7, 7), "a", 0.8)),
            ],
            unmatched_gt=[
                (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
            ],
            unmatched_inf=[
                ScoredLabeledBoundingBox((8, 8), (9, 9), "a", 0.8),
            ],
        ),
        MulticlassInferenceMatches(
            matched=[
                (LabeledBoundingBox((1, 1), (4, 4), "b"), ScoredLabeledBoundingBox((1, 1), (4, 4), "b", 0.9)),
                (LabeledBoundingBox((2, 2), (7, 7), "b"), ScoredLabeledBoundingBox((2, 2), (7, 7), "b", 0.8)),
            ],
            unmatched_gt=[
                (LabeledBoundingBox((1, 1), (2, 2), "b"), None),
            ],
            unmatched_inf=[
                ScoredLabeledBoundingBox((8, 8), (9, 9), "b", 0.7),
            ],
        ),
    ],
    "two single class matchings as IMs": [
        InferenceMatches(
            matched=[
                (LabeledBoundingBox((3, 3), (4, 4), "a"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.9)),
                (LabeledBoundingBox((6, 6), (7, 7), "a"), ScoredLabeledBoundingBox((6, 6), (7, 7), "a", 0.8)),
            ],
            unmatched_gt=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
            ],
            unmatched_inf=[
                ScoredLabeledBoundingBox((8, 8), (9, 9), "a", 0.7),
            ],
        ),
        InferenceMatches(
            matched=[
                (LabeledBoundingBox((1, 1), (4, 4), "b"), ScoredLabeledBoundingBox((1, 1), (4, 4), "b", 0.7)),
                (LabeledBoundingBox((2, 2), (7, 7), "b"), ScoredLabeledBoundingBox((2, 2), (7, 7), "b", 0.6)),
            ],
            unmatched_gt=[
                LabeledBoundingBox((1, 1), (2, 2), "b"),
            ],
            unmatched_inf=[
                ScoredLabeledBoundingBox((8, 8), (9, 9), "b", 0.5),
            ],
        ),
    ],
    "large": [
        MulticlassInferenceMatches(
            matched=[
                (
                    LabeledBoundingBox(top_left=(1, 1), bottom_right=(6, 6), label="cow"),
                    ScoredLabeledBoundingBox((1, 1), (6, 6), "cow", 0.9),
                ),
                (
                    LabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="cow"),
                    ScoredLabeledBoundingBox((10, 10), (20, 20), "cow", 0.75),
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
                        score=0.8,
                    ),
                ),
            ],
            unmatched_inf=[
                ScoredLabeledPolygon(
                    points=[(1, 1), (1, 20), (20, 20), (20, 1)],
                    label="dog",
                    score=0.8,
                ),
            ],
        ),
        MulticlassInferenceMatches(
            matched=[
                (
                    LabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="cow"),
                    ScoredLabeledBoundingBox((10, 10), (20, 20), "cow", 0.77),
                ),
            ],
            unmatched_gt=[
                (
                    LabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="fish"),
                    ScoredLabeledBoundingBox((10, 10), (20, 20), "dog", 0.5),
                ),
            ],
            unmatched_inf=[
                ScoredLabeledBoundingBox((10, 10), (22, 22), "cat", 0.3),
                ScoredLabeledBoundingBox((10, 10), (20, 20), "dog", 0.5),
            ],
        ),
        MulticlassInferenceMatches(
            matched=[
                (
                    LabeledPolygon(
                        points=[(10, 10), (10, 22), (22, 22), (22, 10)],
                        label="cow",
                    ),
                    ScoredLabeledPolygon(
                        points=[(10, 10), (10, 20), (20, 20), (20, 10)],
                        label="cow",
                        score=0.4,
                    ),
                ),
            ],
            unmatched_gt=[],
            unmatched_inf=[
                ScoredLabeledPolygon(
                    points=[(10, 10), (10, 22), (22, 22), (22, 10)],
                    label="cow",
                    score=0.5,
                ),
            ],
        ),
        MulticlassInferenceMatches(
            matched=[
                (
                    LabeledPolygon(
                        points=[(10, 10), (10, 22), (22, 22), (22, 10)],
                        label="cat",
                    ),
                    ScoredLabeledPolygon(
                        points=[(10, 10), (10, 20), (20, 20), (20, 10)],
                        label="cat",
                        score=0.2,
                    ),
                ),
            ],
            unmatched_gt=[],
            unmatched_inf=[
                ScoredLabeledPolygon(
                    points=[(10, 10), (10, 22), (22, 22), (22, 10)],
                    label="dog",
                    score=0.1,
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
                    score=0.99,
                ),
            ],
        ),
    ],
    "only tps": [
        MulticlassInferenceMatches(
            matched=[
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.3)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.2)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.1)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.01)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.7)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.6)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.5)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.4)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.3)),
                (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.3)),
                (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2)),
                (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.1)),
                (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.01)),
                (LabeledBoundingBox((1, 1), (2, 2), "d"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.99)),
                (LabeledBoundingBox((1, 1), (2, 2), "d"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.9)),
                (LabeledBoundingBox((1, 1), (2, 2), "d"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.8)),
                (LabeledBoundingBox((1, 1), (2, 2), "d"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.7)),
                (LabeledBoundingBox((1, 1), (2, 2), "d"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.6)),
                (LabeledBoundingBox((1, 1), (2, 2), "e"), ScoredLabeledBoundingBox((1, 1), (2, 2), "e", 0.99)),
                (LabeledBoundingBox((1, 1), (2, 2), "e"), ScoredLabeledBoundingBox((1, 1), (2, 2), "e", 0.9)),
                (LabeledBoundingBox((1, 1), (2, 2), "e"), ScoredLabeledBoundingBox((1, 1), (2, 2), "e", 0.01)),
            ],
            unmatched_gt=[],
            unmatched_inf=[],
        ),
    ],
    "tps and fps and fns": [
        MulticlassInferenceMatches(
            matched=[
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.3)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.2)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.1)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.01)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.7)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.6)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.5)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.4)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.3)),
                (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.3)),
                (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2)),
                (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.1)),
                (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.01)),
                (LabeledBoundingBox((1, 1), (2, 2), "d"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.99)),
                (LabeledBoundingBox((1, 1), (2, 2), "d"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.9)),
                (LabeledBoundingBox((1, 1), (2, 2), "d"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.8)),
                (LabeledBoundingBox((1, 1), (2, 2), "d"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.7)),
                (LabeledBoundingBox((1, 1), (2, 2), "d"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.6)),
                (LabeledBoundingBox((1, 1), (2, 2), "e"), ScoredLabeledBoundingBox((1, 1), (2, 2), "e", 0.99)),
                (LabeledBoundingBox((1, 1), (2, 2), "e"), ScoredLabeledBoundingBox((1, 1), (2, 2), "e", 0.9)),
                (LabeledBoundingBox((1, 1), (2, 2), "e"), ScoredLabeledBoundingBox((1, 1), (2, 2), "e", 0.01)),
            ],
            unmatched_gt=[
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8)),
                (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.7)),
                (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2)),
                (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.1)),
                (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.1)),
                (LabeledBoundingBox((1, 1), (2, 2), "e"), None),
                (LabeledBoundingBox((1, 1), (2, 2), "e"), None),
                (LabeledBoundingBox((1, 1), (2, 2), "e"), None),
                (LabeledBoundingBox((1, 1), (2, 2), "e"), None),
            ],
            unmatched_inf=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.7),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.1),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.1),
            ],
        ),
    ],
}


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name",
    [
        "zeros with unmatched gt and unmatched inf",
        "zeros with two matchings",
        "zeros, but b is confused with a",
        "zeros, but a is confused with b",
        "only confusion",
    ],
)
def test__no_curve_plot(
    test_name: str,
) -> None:
    from kolena._experimental.object_detection.utils import compute_f1_plot
    from kolena._experimental.object_detection.utils import compute_pr_plot
    from kolena._experimental.object_detection.utils import compute_pr_curve

    f1: CurvePlot = compute_f1_plot(all_matches=TEST_MATCHING[test_name], curve_label=test_name)
    pr: CurvePlot = compute_pr_plot(all_matches=TEST_MATCHING[test_name], curve_label=test_name)
    pr_curve: Curve = compute_pr_curve(all_matches=TEST_MATCHING[test_name], curve_label=test_name)
    assert f1 is None
    assert pr is None
    assert pr_curve is None


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, f1_curve, pr_curve",
    [
        (
            "zeros, but one match for label a",
            None,
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[1 / 3, 0],
                        y=[1, 1],
                        label="zeros, but one match for label a",
                        extra={"F1": [0.5, 0.5], "Threshold": [0.0, 0.0]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "zeros, but one match for label b",
            None,
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[1 / 3, 0],
                        y=[1, 1],
                        label="zeros, but one match for label b",
                        extra={"F1": [0.5, 0.5], "Threshold": [0.0, 0.0]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "only confusion, one TP for a",
            None,
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[1 / 3, 0.0],
                        y=[1.0, 1.0],
                        label="only confusion, one TP for a",
                        extra={"F1": [0.5, 0.5], "Threshold": [0.9, 0.9]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "only confusion, one TP for b",
            None,
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[1 / 3, 0.0],
                        y=[1.0, 1.0],
                        label="only confusion, one TP for b",
                        extra={"F1": [0.5, 0.5], "Threshold": [0.9, 0.9]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
    ],
)
def test__no_curve_plot_only_confusion(
    test_name: str,
    f1_curve: CurvePlot,
    pr_curve: CurvePlot,
) -> None:
    from kolena._experimental.object_detection.utils import compute_f1_plot
    from kolena._experimental.object_detection.utils import compute_pr_plot
    from kolena._experimental.object_detection.utils import compute_pr_curve

    f1: CurvePlot = compute_f1_plot(all_matches=TEST_MATCHING[test_name], curve_label=test_name)
    pr: CurvePlot = compute_pr_plot(all_matches=TEST_MATCHING[test_name], curve_label=test_name)
    single_pr_curve: Curve = compute_pr_curve(all_matches=TEST_MATCHING[test_name], curve_label=test_name)

    assert f1 == f1_curve
    assert pr == pr_curve
    assert single_pr_curve is None and pr is None or single_pr_curve == pr_curve.curves[0]


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, f1_curve, pr_curve",
    [
        (
            "no confusion, one TP per label",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.5, 0.6],
                        y=[2 / 3, 0.4],
                        label="no confusion, one TP per label",
                        extra={"Precision": [1.0, 1.0], "Recall": [0.5, 0.25]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[0.5, 0.25, 0],
                        y=[1, 1, 1],
                        label="no confusion, one TP per label",
                        extra={"F1": [2 / 3, 0.4, 0.4], "Threshold": [0.5, 0.6, 0.6]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "ones",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.8, 0.9],
                        y=[2 / 3, 0.4],
                        label="ones",
                        extra={"Precision": [1.0, 1.0], "Recall": [0.5, 0.25]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[0.5, 0.25, 0.0],
                        y=[1.0, 1.0, 1.0],
                        label="ones",
                        extra={"F1": [2 / 3, 0.4, 0.4], "Threshold": [0.8, 0.9, 0.9]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "ones, with two matchings, TPs",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.8, 0.9],
                        y=[2 / 3, 0.4],
                        label="ones, with two matchings, TPs",
                        extra={"Precision": [1.0, 1.0], "Recall": [0.5, 0.25]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[0.5, 0.25, 0.0],
                        y=[1.0, 1.0, 1.0],
                        label="ones, with two matchings, TPs",
                        extra={"F1": [2 / 3, 0.4, 0.4], "Threshold": [0.8, 0.9, 0.9]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "ones, with two matchings, mixed",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.6, 0.9],
                        y=[4 / 7, 0.4],
                        label="ones, with two matchings, mixed",
                        extra={"Precision": [2 / 3, 1.0], "Recall": [0.5, 0.25]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[0.5, 0.25, 0.0],
                        y=[2 / 3, 1.0, 1.0],
                        label="ones, with two matchings, mixed",
                        extra={"F1": [4 / 7, 0.4, 0.4], "Threshold": [0.6, 0.9, 0.9]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "two single class matchings",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.8, 0.9],
                        y=[8 / 11, 0.5],
                        label="two single class matchings",
                        extra={"Precision": [0.8, 1.0], "Recall": [2 / 3, 1 / 3]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[2 / 3, 1 / 3, 0.0],
                        y=[0.8, 1.0, 1.0],
                        label="two single class matchings",
                        extra={"F1": [8 / 11, 0.5, 0.5], "Threshold": [0.8, 0.9, 0.9]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "two single class matchings as IMs",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.6, 0.7, 0.8, 0.9],
                        y=[8 / 11, 0.6, 0.5, 2 / 7],
                        label="two single class matchings as IMs",
                        extra={"Precision": [0.8, 0.75, 1.0, 1.0], "Recall": [2 / 3, 0.5, 1 / 3, 1 / 6]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[2 / 3, 0.5, 1 / 3, 1 / 6, 0.0],
                        y=[0.8, 0.75, 1.0, 1.0, 1.0],
                        label="two single class matchings as IMs",
                        extra={"F1": [8 / 11, 0.6, 0.5, 2 / 7, 2 / 7], "Threshold": [0.6, 0.7, 0.8, 0.9, 0.9]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "large",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.2, 0.4, 0.75, 0.77, 0.9, 0.99],
                        y=[10 / 19, 8 / 17, 3 / 7, 4 / 13, 2 / 11, 0.0],
                        label="large",
                        extra={
                            "Precision": [0.5, 0.5, 0.6, 0.5, 0.5, 0.0],
                            "Recall": [5 / 9, 4 / 9, 1 / 3, 2 / 9, 1 / 9, 0.0],
                        },
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[5 / 9, 4 / 9, 1 / 3, 2 / 9, 1 / 9, 0.0],
                        y=[0.5, 0.5, 0.6, 0.5, 0.5, 0.0],
                        label="large",
                        extra={
                            "F1": [10 / 19, 8 / 17, 3 / 7, 4 / 13, 2 / 11, 0.0],
                            "Threshold": [0.2, 0.4, 0.75, 0.77, 0.9, 0.99],
                        },
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "only tps",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99],
                        y=[1.0, 44 / 47, 8 / 9, 36 / 43, 3 / 4, 28 / 39, 13 / 19, 11 / 18, 9 / 17, 12 / 31, 3 / 14],
                        label="only tps",
                        extra={
                            "Precision": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            "Recall": [1.0, 0.88, 0.8, 0.72, 0.6, 0.56, 0.52, 0.44, 0.36, 0.24, 0.12],
                        },
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[1.0, 0.88, 0.8, 0.72, 0.6, 0.56, 0.52, 0.44, 0.36, 0.24, 0.12, 0.0],
                        y=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        label="only tps",
                        extra={
                            "F1": [
                                1.0,
                                44 / 47,
                                8 / 9,
                                36 / 43,
                                3 / 4,
                                28 / 39,
                                13 / 19,
                                11 / 18,
                                9 / 17,
                                12 / 31,
                                3 / 14,
                                0.21428571428571425,
                            ],
                            "Threshold": [0.01, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.99],
                        },
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "tps and fps and fns",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[
                            0.01,
                            0.1,
                            0.2,
                            0.3,
                            0.4,
                            0.5,
                            0.6,
                            0.7,
                            0.8,
                            0.9,
                            0.99,
                        ],
                        y=[
                            25 / 37,
                            44 / 71,
                            20 / 31,
                            36 / 59,
                            15 / 28,
                            28 / 55,
                            13 / 27,
                            11 / 26,
                            18 / 49,
                            12 / 41,
                            3 / 19,
                        ],
                        label="tps and fps and fns",
                        extra={
                            "Precision": [
                                25 / 39,
                                11 / 18,
                                20 / 27,
                                0.75,
                                5 / 7,
                                7 / 10,
                                13 / 19,
                                11 / 17,
                                9 / 14,
                                1,
                                1,
                            ],
                            "Recall": [
                                5 / 7,
                                22 / 35,
                                4 / 7,
                                18 / 35,
                                3 / 7,
                                0.4,
                                13 / 35,
                                11 / 35,
                                9 / 35,
                                6 / 35,
                                3 / 35,
                            ],
                        },
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[
                            5 / 7,
                            22 / 35,
                            4 / 7,
                            18 / 35,
                            3 / 7,
                            0.4,
                            13 / 35,
                            11 / 35,
                            9 / 35,
                            6 / 35,
                            3 / 35,
                            0,
                        ],
                        y=[
                            25 / 39,
                            11 / 18,
                            20 / 27,
                            0.75,
                            5 / 7,
                            7 / 10,
                            13 / 19,
                            11 / 17,
                            9 / 14,
                            1,
                            1,
                            1,
                        ],
                        label="tps and fps and fns",
                        extra={
                            "F1": [
                                25 / 37,
                                44 / 71,
                                20 / 31,
                                36 / 59,
                                15 / 28,
                                28 / 55,
                                13 / 27,
                                11 / 26,
                                18 / 49,
                                12 / 41,
                                3 / 19,
                                3 / 19,
                            ],
                            "Threshold": [
                                0.01,
                                0.1,
                                0.2,
                                0.3,
                                0.4,
                                0.5,
                                0.6,
                                0.7,
                                0.8,
                                0.9,
                                0.99,
                                0.99,
                            ],
                        },
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
    ],
)
def test__curve_plots(
    test_name: str,
    f1_curve: CurvePlot,
    pr_curve: CurvePlot,
) -> None:
    from kolena._experimental.object_detection.utils import compute_f1_plot
    from kolena._experimental.object_detection.utils import compute_pr_plot
    from kolena._experimental.object_detection.utils import compute_pr_curve

    f1: CurvePlot = compute_f1_plot(all_matches=TEST_MATCHING[test_name], curve_label=test_name)
    pr: CurvePlot = compute_pr_plot(all_matches=TEST_MATCHING[test_name], curve_label=test_name)
    single_pr_curve = compute_pr_curve(all_matches=TEST_MATCHING[test_name], curve_label=test_name)
    assert_curve_plot_equals_expected(f1, f1_curve)
    assert_curve_plot_equals_expected(pr, pr_curve)
    assert_curves([pr.curves[0]], [single_pr_curve])


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, f1_curve, pr_curve",
    [
        (
            "no confusion, one TP per label",
            None,
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5, 0.0], y=[1.0, 1.0], label="a", extra={"F1": [2 / 3, 2 / 3], "Threshold": [0.6, 0.6]}),
                    Curve(x=[0.5, 0.0], y=[1.0, 1.0], label="b", extra={"F1": [2 / 3, 2 / 3], "Threshold": [0.5, 0.5]}),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "only confusion",
            None,
            None,
        ),
        (
            "only confusion, one TP for a",
            None,
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5, 0.0], y=[1.0, 1.0], label="a", extra={"F1": [2 / 3, 2 / 3], "Threshold": [0.9, 0.9]}),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "only confusion, one TP for b",
            None,
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5, 0.0], y=[1.0, 1.0], label="b", extra={"F1": [2 / 3, 2 / 3], "Threshold": [0.9, 0.9]}),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "ones",
            None,
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5, 0.0], y=[1.0, 1.0], label="a", extra={"F1": [2 / 3, 2 / 3], "Threshold": [0.9, 0.9]}),
                    Curve(x=[0.5, 0.0], y=[1.0, 1.0], label="b", extra={"F1": [2 / 3, 2 / 3], "Threshold": [0.8, 0.8]}),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "ones, with two matchings, TPs",
            None,
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5, 0.0], y=[1.0, 1.0], label="a", extra={"F1": [2 / 3, 2 / 3], "Threshold": [0.9, 0.9]}),
                    Curve(x=[0.5, 0.0], y=[1.0, 1.0], label="b", extra={"F1": [2 / 3, 2 / 3], "Threshold": [0.8, 0.8]}),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
    ],
)
def test__curve_plots__multiclass__unique(
    test_name: str,
    f1_curve: CurvePlot,
    pr_curve: CurvePlot,
) -> None:
    from kolena._experimental.object_detection.utils import compute_f1_plot_multiclass
    from kolena._experimental.object_detection.utils import compute_pr_plot_multiclass

    f1: CurvePlot = compute_f1_plot_multiclass(all_matches=TEST_MATCHING[test_name])
    pr: CurvePlot = compute_pr_plot_multiclass(all_matches=TEST_MATCHING[test_name])
    assert f1 == f1_curve
    assert pr == pr_curve


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, f1_curve, pr_curve",
    [
        (
            "ones, with two matchings, mixed",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold Per Class",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.6, 0.8], y=[0.5, 0.0], label="b", extra={"Precision": [0.5, 0.0], "Recall": [0.5, 0.0]}),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5, 0.0], y=[1.0, 1.0], label="a", extra={"F1": [2 / 3, 2 / 3], "Threshold": [0.9, 0.9]}),
                    Curve(x=[0.5, 0.0], y=[0.5, 0.0], label="b", extra={"F1": [0.5, 0.0], "Threshold": [0.6, 0.8]}),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "two single class matchings",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold Per Class",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.8, 0.9],
                        y=[2 / 3, 0.5],
                        label="a",
                        extra={"Precision": [2 / 3, 1.0], "Recall": [2 / 3, 1 / 3]},
                    ),
                    Curve(
                        x=[0.8, 0.9],
                        y=[0.8, 0.5],
                        label="b",
                        extra={"Precision": [1.0, 1.0], "Recall": [2 / 3, 1 / 3]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[2 / 3, 1 / 3, 0.0],
                        y=[2 / 3, 1.0, 1.0],
                        label="a",
                        extra={"F1": [2 / 3, 0.5, 0.5], "Threshold": [0.8, 0.9, 0.9]},
                    ),
                    Curve(
                        x=[2 / 3, 1 / 3, 0.0],
                        y=[1.0, 1.0, 1.0],
                        label="b",
                        extra={"F1": [0.8, 0.5, 0.5], "Threshold": [0.8, 0.9, 0.9]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "large",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold Per Class",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.2, 0.3],
                        y=[2 / 3, 0.0],
                        label="cat",
                        extra={"Precision": [0.5, 0.0], "Recall": [1.0, 0.0]},
                    ),
                    Curve(
                        x=[0.4, 0.75, 0.77, 0.9],
                        y=[2 / 3, 0.6, 4 / 9, 0.25],
                        label="cow",
                        extra={"Precision": [0.8, 1.0, 1.0, 1.0], "Recall": [4 / 7, 3 / 7, 2 / 7, 1 / 7]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[1.0, 0.0], y=[0.5, 0.0], label="cat", extra={"F1": [2 / 3, 0.0], "Threshold": [0.2, 0.3]}),
                    Curve(
                        x=[4 / 7, 3 / 7, 2 / 7, 1 / 7, 0.0],
                        y=[0.8, 1.0, 1.0, 1.0, 1.0],
                        label="cow",
                        extra={"F1": [2 / 3, 0.6, 4 / 9, 0.25, 0.25], "Threshold": [0.4, 0.75, 0.77, 0.9, 0.9]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "only tps",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold Per Class",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.01, 0.1, 0.2, 0.3, 0.8, 0.9, 0.99],
                        y=[1.0, 12 / 13, 5 / 6, 8 / 11, 0.6, 4 / 9, 0.25],
                        label="a",
                        extra={
                            "Precision": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            "Recall": [1.0, 6 / 7, 5 / 7, 4 / 7, 3 / 7, 2 / 7, 1 / 7],
                        },
                    ),
                    Curve(
                        x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        y=[1.0, 10 / 11, 0.8, 2 / 3, 0.5, 2 / 7],
                        label="b",
                        extra={
                            "Precision": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                            "Recall": [1.0, 5 / 6, 2 / 3, 0.5, 1 / 3, 1 / 6],
                        },
                    ),
                    Curve(
                        x=[0.01, 0.1, 0.2, 0.3],
                        y=[1.0, 6 / 7, 2 / 3, 0.4],
                        label="c",
                        extra={"Precision": [1.0, 1.0, 1.0, 1.0], "Recall": [1.0, 0.75, 0.5, 0.25]},
                    ),
                    Curve(
                        x=[0.6, 0.7, 0.8, 0.9, 0.99],
                        y=[1.0, 8 / 9, 0.75, 4 / 7, 1 / 3],
                        label="d",
                        extra={"Precision": [1.0, 1.0, 1.0, 1.0, 1.0], "Recall": [1.0, 0.8, 0.6, 0.4, 0.2]},
                    ),
                    Curve(
                        x=[0.01, 0.9, 0.99],
                        y=[1.0, 0.8, 0.5],
                        label="e",
                        extra={"Precision": [1.0, 1.0, 1.0], "Recall": [1.0, 2 / 3, 1 / 3]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[1.0, 6 / 7, 5 / 7, 4 / 7, 3 / 7, 2 / 7, 1 / 7, 0.0],
                        y=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        label="a",
                        extra={
                            "F1": [1.0, 12 / 13, 5 / 6, 8 / 11, 0.6, 4 / 9, 0.25, 0.25],
                            "Threshold": [0.01, 0.1, 0.2, 0.3, 0.8, 0.9, 0.99, 0.99],
                        },
                    ),
                    Curve(
                        x=[1.0, 5 / 6, 2 / 3, 0.5, 1 / 3, 1 / 6, 0.0],
                        y=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        label="b",
                        extra={
                            "F1": [1.0, 10 / 11, 0.8, 2 / 3, 0.5, 2 / 7, 2 / 7],
                            "Threshold": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8],
                        },
                    ),
                    Curve(
                        x=[1.0, 0.75, 0.5, 0.25, 0.0],
                        y=[1.0, 1.0, 1.0, 1.0, 1.0],
                        label="c",
                        extra={"F1": [1.0, 6 / 7, 2 / 3, 0.4, 0.4], "Threshold": [0.01, 0.1, 0.2, 0.3, 0.3]},
                    ),
                    Curve(
                        x=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                        y=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        label="d",
                        extra={
                            "F1": [1.0, 8 / 9, 0.75, 4 / 7, 1 / 3, 1 / 3],
                            "Threshold": [0.6, 0.7, 0.8, 0.9, 0.99, 0.99],
                        },
                    ),
                    Curve(
                        x=[1.0, 2 / 3, 1 / 3, 0.0],
                        y=[1.0, 1.0, 1.0, 1.0],
                        label="e",
                        extra={"F1": [1.0, 0.8, 0.5, 0.5], "Threshold": [0.01, 0.9, 0.99, 0.99]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "tps and fps and fns",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold Per Class",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.01, 0.1, 0.2, 0.3, 0.8, 0.9, 0.99],
                        y=[0.7, 12 / 19, 5 / 9, 8 / 17, 0.375, 4 / 11, 0.2],
                        label="a",
                        extra={
                            "Precision": [7 / 11, 0.6, 5 / 9, 0.5, 3 / 7, 1.0, 1.0],
                            "Recall": [7 / 9, 2 / 3, 5 / 9, 4 / 9, 1 / 3, 2 / 9, 1 / 9],
                        },
                    ),
                    Curve(
                        x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        y=[0.8, 5 / 7, 8 / 13, 0.5, 4 / 11, 0.2],
                        label="b",
                        extra={
                            "Precision": [6 / 7, 5 / 6, 0.8, 0.75, 2 / 3, 0.5],
                            "Recall": [0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
                        },
                    ),
                    Curve(
                        x=[0.01, 0.1, 0.2, 0.3, 0.7],
                        y=[2 / 3, 6 / 11, 0.4, 0.25, 0.0],
                        label="c",
                        extra={"Precision": [2 / 3, 0.6, 0.5, 0.5, 0.0], "Recall": [2 / 3, 0.5, 1 / 3, 1 / 6, 0.0]},
                    ),
                    Curve(
                        x=[0.6, 0.7, 0.8, 0.9, 0.99],
                        y=[1.0, 8 / 9, 0.75, 4 / 7, 1 / 3],
                        label="d",
                        extra={"Precision": [1.0, 1.0, 1.0, 1.0, 1.0], "Recall": [1.0, 0.8, 0.6, 0.4, 0.2]},
                    ),
                    Curve(
                        x=[0.01, 0.9, 0.99],
                        y=[0.6, 4 / 9, 0.25],
                        label="e",
                        extra={"Precision": [1.0, 1.0, 1.0], "Recall": [3 / 7, 2 / 7, 1 / 7]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[7 / 9, 2 / 3, 5 / 9, 4 / 9, 1 / 3, 2 / 9, 1 / 9, 0.0],
                        y=[7 / 11, 0.6, 5 / 9, 0.5, 3 / 7, 1.0, 1.0, 1.0],
                        label="a",
                        extra={
                            "F1": [0.7, 12 / 19, 5 / 9, 8 / 17, 0.375, 4 / 11, 0.2, 0.2],
                            "Threshold": [0.01, 0.1, 0.2, 0.3, 0.8, 0.9, 0.99, 0.99],
                        },
                    ),
                    Curve(
                        x=[0.75, 0.625, 0.5, 0.375, 0.25, 0.125, 0.0],
                        y=[6 / 7, 5 / 6, 0.8, 0.75, 2 / 3, 0.5, 0.5],
                        label="b",
                        extra={
                            "F1": [0.8, 5 / 7, 8 / 13, 0.5, 4 / 11, 0.2, 0.2],
                            "Threshold": [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.8],
                        },
                    ),
                    Curve(
                        x=[2 / 3, 0.5, 1 / 3, 1 / 6, 0.0],
                        y=[2 / 3, 0.6, 0.5, 0.5, 0.0],
                        label="c",
                        extra={"F1": [2 / 3, 6 / 11, 0.4, 0.25, 0.0], "Threshold": [0.01, 0.1, 0.2, 0.3, 0.7]},
                    ),
                    Curve(
                        x=[1.0, 0.8, 0.6, 0.4, 0.2, 0.0],
                        y=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        label="d",
                        extra={
                            "F1": [1.0, 8 / 9, 0.75, 4 / 7, 1 / 3, 1 / 3],
                            "Threshold": [0.6, 0.7, 0.8, 0.9, 0.99, 0.99],
                        },
                    ),
                    Curve(
                        x=[3 / 7, 2 / 7, 1 / 7, 0.0],
                        y=[1.0, 1.0, 1.0, 1.0],
                        label="e",
                        extra={"F1": [0.6, 4 / 9, 0.25, 0.25], "Threshold": [0.01, 0.9, 0.99, 0.99]},
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
    ],
)
def test__curve_plots__multiclass(
    test_name: str,
    f1_curve: CurvePlot,
    pr_curve: CurvePlot,
) -> None:
    from kolena._experimental.object_detection.utils import compute_f1_plot_multiclass
    from kolena._experimental.object_detection.utils import compute_pr_plot_multiclass

    f1: CurvePlot = compute_f1_plot_multiclass(all_matches=TEST_MATCHING[test_name])
    pr: CurvePlot = compute_pr_plot_multiclass(all_matches=TEST_MATCHING[test_name])
    assert_curve_plot_equals_expected(f1, f1_curve)
    assert_curve_plot_equals_expected(pr, pr_curve)


@pytest.mark.metrics
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
            "zeros, but one match for label a",
            TEST_MATCHING["zeros, but one match for label a"],
            ["a", "b"],
            [
                [1, 0],
                [0, 0],
            ],
        ),
        (
            "zeros, but one match for label b",
            TEST_MATCHING["zeros, but one match for label b"],
            ["a", "b"],
            [
                [0, 0],
                [0, 1],
            ],
        ),
        (
            "zeros, but b is confused with a",
            TEST_MATCHING["zeros, but b is confused with a"],
            ["a", "b"],
            [
                [0, 1],
                [0, 0],
            ],
        ),
        (
            "zeros, but a is confused with b",
            TEST_MATCHING["zeros, but a is confused with b"],
            ["a", "b"],
            [
                [0, 0],
                [1, 0],
            ],
        ),
        (
            "no confusion, one TP per label",
            TEST_MATCHING["no confusion, one TP per label"],
            ["a", "b"],
            [
                [1, 0],
                [0, 1],
            ],
        ),
        (
            "only confusion",
            TEST_MATCHING["only confusion"],
            ["a", "b"],
            [
                [0, 1],
                [1, 0],
            ],
        ),
        (
            "only confusion, one TP for a",
            TEST_MATCHING["only confusion, one TP for a"],
            ["a", "b"],
            [
                [1, 1],
                [1, 0],
            ],
        ),
        (
            "only confusion, one TP for b",
            TEST_MATCHING["only confusion, one TP for b"],
            ["a", "b"],
            [
                [0, 1],
                [1, 1],
            ],
        ),
        (
            "ones",
            TEST_MATCHING["ones"],
            ["a", "b"],
            [
                [1, 1],
                [1, 1],
            ],
        ),
        (
            "ones, with two matchings, TPs",
            TEST_MATCHING["ones, with two matchings, TPs"],
            ["a", "b"],
            [
                [1, 1],
                [1, 1],
            ],
        ),
        (
            "ones, with two matchings, mixed",
            TEST_MATCHING["ones, with two matchings, mixed"],
            ["a", "b"],
            [
                [1, 1],
                [1, 1],
            ],
        ),
        (
            "two single class matchings",
            TEST_MATCHING["two single class matchings"],
            ["a", "b"],
            [
                [2, 0],
                [0, 2],
            ],
        ),
        (
            "large",
            TEST_MATCHING["large"],
            ["cat", "cow", "dog", "fish"],
            [
                [1, 0, 0, 0],
                [1, 4, 2, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0],
            ],
        ),
        (
            "extra unmatched inf",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), None),
                    ],
                    unmatched_inf=[ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0)],
                ),
            ],
            ["a", "b"],
            [
                [1, 0],
                [0, 0],
            ],
        ),
    ],
)
def test__confusion_matrix(
    test_name: str,
    matchings: List[MulticlassInferenceMatches],
    ordered_labels: List[str],
    matrix: List[List[int]],
) -> None:
    from kolena._experimental.object_detection.utils import compute_confusion_matrix_plot

    conf_mat = compute_confusion_matrix_plot(all_matches=matchings, plot_title=test_name)
    assert conf_mat == ConfusionMatrix(title=test_name, labels=ordered_labels, matrix=matrix)


@pytest.mark.metrics
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
        ),
        (
            "zeros with unmatched gt and unmatched inf",
            TEST_MATCHING["zeros with unmatched gt and unmatched inf"],
        ),
        (
            "zeros with two matchings",
            TEST_MATCHING["zeros with two matchings"],
        ),
    ],
)
def test__confusion_matrix_fails(
    test_name: str,
    matchings: List[MulticlassInferenceMatches],
) -> None:
    from kolena._experimental.object_detection.utils import compute_confusion_matrix_plot

    conf_mat = compute_confusion_matrix_plot(all_matches=matchings, plot_title=test_name)
    assert conf_mat is None
