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
from typing import Union

import pytest

from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledPolygon
from kolena.workflow.evaluator import ConfusionMatrix
from kolena.workflow.evaluator import Curve
from kolena.workflow.evaluator import CurvePlot
from kolena.workflow.metrics import InferenceMatches
from kolena.workflow.metrics import MulticlassInferenceMatches


TOLERANCE = 1e-8


def assert_curve_equal(c1: Curve, c2: Curve) -> None:
    assert type(c1) == type(c2)
    c1_keys = ["x", "y", "label"]
    for k in c1_keys:
        v1 = getattr(c1, k)
        v2 = getattr(c2, k)
        if k == "label":
            assert v1 == v2
        else:
            assert all(val1 == pytest.approx(val2, TOLERANCE) for val1, val2 in zip(v1, v2))


def assert_curveplot_equal(c1: CurvePlot, c2: CurvePlot):
    assert c1.title == c2.title
    assert c1.x_config == c2.x_config
    assert c1.y_config == c2.y_config
    assert c1.x_label == c2.x_label
    assert c1.y_label == c2.y_label
    assert len(c1.curves) == len(c2.curves)
    for c1, c2 in zip(c1.curves, c2.curves):
        assert_curve_equal(c1, c2)


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, matchings, f1_curve, pr_curve",
    [
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
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[0.0], label="zeros with unmatched gt and unmatched inf"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.0], y=[0.0], label="zeros with unmatched gt and unmatched inf"),
                ],
                x_config=None,
                y_config=None,
            ),
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
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[0.0], label="zeros with two matchings"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.0], y=[0.0], label="zeros with two matchings"),
                ],
                x_config=None,
                y_config=None,
            ),
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
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[0.5], label="zeros, but one match for label a"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[1 / 3], y=[1.0], label="zeros, but one match for label a"),
                ],
                x_config=None,
                y_config=None,
            ),
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
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[0.5], label="zeros, but one match for label b"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[1 / 3], y=[1.0], label="zeros, but one match for label b"),
                ],
                x_config=None,
                y_config=None,
            ),
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
                    unmatched_inf=[ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0)],
                ),
            ],
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[0.0], label="zeros, but b is confused with a"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.0], y=[0.0], label="zeros, but b is confused with a"),
                ],
                x_config=None,
                y_config=None,
            ),
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
                    unmatched_inf=[ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0)],
                ),
            ],
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[0.0], label="zeros, but a is confused with b"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.0], y=[0.0], label="zeros, but a is confused with b"),
                ],
                x_config=None,
                y_config=None,
            ),
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
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[2 / 3], label="no confusion, one TP per label"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[1 / 2], y=[1.0], label="no confusion, one TP per label"),
                ],
                x_config=None,
                y_config=None,
            ),
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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0),
                        ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0),
                    ],
                ),
            ],
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[0.0], label="only confusion"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.0], y=[0.0], label="only confusion"),
                ],
                x_config=None,
                y_config=None,
            ),
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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0),
                        ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0),
                    ],
                ),
            ],
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[1 / 3], label="only confusion, one TP for a"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[1 / 3], y=[1 / 3], label="only confusion, one TP for a"),
                ],
                x_config=None,
                y_config=None,
            ),
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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0),
                        ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0),
                    ],
                ),
            ],
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[1 / 3], label="only confusion, one TP for b"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[1 / 3], y=[1 / 3], label="only confusion, one TP for b"),
                ],
                x_config=None,
                y_config=None,
            ),
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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0),
                        ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0),
                    ],
                ),
            ],
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[0.5], label="ones"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5], y=[0.5], label="ones"),
                ],
                x_config=None,
                y_config=None,
            ),
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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0),
                        ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0),
                    ],
                ),
            ],
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[0.5], label="ones, with two matchings, TPs"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5], y=[0.5], label="ones, with two matchings, TPs"),
                ],
                x_config=None,
                y_config=None,
            ),
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
                    unmatched_inf=[ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0)],
                ),
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((7, 7), (8, 8), "b"), ScoredLabeledBoundingBox((7, 7), (8, 8), "b", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                    ],
                    unmatched_inf=[ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)],
                ),
            ],
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.0], y=[0.5], label="ones, with two matchings, mixed"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5], y=[0.5], label="ones, with two matchings, mixed"),
                ],
                x_config=None,
                y_config=None,
            ),
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
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.0],
                        y=[2 / 3],
                        label="two single class matchings",
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
                        x=[2 / 3],
                        y=[2 / 3],
                        label="two single class matchings",
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "two single class matchings as IMs",
            [
                InferenceMatches(
                    matched=[
                        (LabeledBoundingBox((3, 3), (4, 4), "a"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                        (LabeledBoundingBox((6, 6), (7, 7), "a"), ScoredLabeledBoundingBox((6, 6), (7, 7), "a", 0)),
                    ],
                    unmatched_gt=[
                        LabeledBoundingBox((1, 1), (2, 2), "a"),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "a", 0),
                    ],
                ),
                InferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (4, 4), "b"), ScoredLabeledBoundingBox((1, 1), (4, 4), "b", 0)),
                        (LabeledBoundingBox((2, 2), (7, 7), "b"), ScoredLabeledBoundingBox((2, 2), (7, 7), "b", 0)),
                    ],
                    unmatched_gt=[
                        LabeledBoundingBox((1, 1), (2, 2), "b"),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((8, 8), (9, 9), "b", 0),
                    ],
                ),
            ],
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[0.0],
                        y=[2 / 3],
                        label="two single class matchings as IMs",
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
                        x=[2 / 3],
                        y=[2 / 3],
                        label="two single class matchings as IMs",
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
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
                            ScoredLabeledBoundingBox((10.0, 10.0), (20.0, 20.0), "cow", 0.75),
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
                            LabeledBoundingBox(top_left=(10.0, 10.0), bottom_right=(22.0, 22.0), label="cow"),
                            ScoredLabeledBoundingBox((10.0, 10.0), (20.0, 20.0), "cow", 0.77),
                        ),
                    ],
                    unmatched_gt=[
                        (
                            LabeledBoundingBox(top_left=(10, 10), bottom_right=(22, 22), label="fish"),
                            ScoredLabeledBoundingBox((10, 10), (20, 20), "dog", 0.5),
                        ),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((10.0, 10.0), (22.0, 22.0), "cat", 0.3),
                        ScoredLabeledBoundingBox((10, 10), (20, 20), "dog", 0.5),
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
                                score=0.4,
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
                                score=0.2,
                            ),
                        ),
                    ],
                    unmatched_gt=[],
                    unmatched_inf=[
                        ScoredLabeledPolygon(
                            points=[(10.0, 10.0), (10.0, 22.0), (22.0, 22.0), (22.0, 10.0)],
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
                            ScoredLabeledPolygon(
                                points=[(10, 10), (10, 20), (20, 20), (20, 10)],
                                label="cat",
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
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(
                        x=[
                            0.1,
                            0.2,
                            0.3,
                            0.4,
                            0.5,
                            0.75,
                            0.77,
                            0.8,
                            0.9,
                            0.99,
                        ],
                        y=[
                            0.5,
                            0.5263157894736842,
                            0.4444444444444444,
                            0.47058823529411764,
                            0.375,
                            0.42857142857142855,
                            0.30769230769230765,
                            0.16666666666666666,
                            0.1818181818181818,
                            0.0,
                        ],
                        label="large",
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
                            0.5555555555555556,
                            0.5555555555555556,
                            0.4444444444444444,
                            0.4444444444444444,
                            0.3333333333333333,
                            0.3333333333333333,
                            0.2222222222222222,
                            0.1111111111111111,
                            0.1111111111111111,
                            0.0,
                        ],
                        y=[
                            0.45454545454545453,
                            0.5,
                            0.4444444444444444,
                            0.5,
                            0.42857142857142855,
                            0.6,
                            0.5,
                            0.3333333333333333,
                            0.5,
                            0.0,
                        ],
                        label="large",
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "only tps",
            [
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
                            1.0,
                            0.9361702127659575,
                            0.888888888888889,
                            0.8372093023255813,
                            0.7499999999999999,
                            0.717948717948718,
                            0.6842105263157895,
                            0.6111111111111112,
                            0.5294117647058824,
                            0.3870967741935484,
                            0.21428571428571425,
                        ],
                        label="only tps",
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
                            1.0,
                            0.88,
                            0.8,
                            0.72,
                            0.6,
                            0.56,
                            0.52,
                            0.44,
                            0.36,
                            0.24,
                            0.12,
                        ],
                        y=[
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                            1.0,
                        ],
                        label="only tps",
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "tps and fps and fns",
            [
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
                            0.6756756756756757,
                            0.619718309859155,
                            0.6451612903225806,
                            0.6101694915254237,
                            0.5357142857142858,
                            0.509090909090909,
                            0.4814814814814815,
                            0.4230769230769231,
                            0.36734693877551017,
                            0.2926829268292683,
                            0.15789473684210528,
                        ],
                        label="tps and fps and fns",
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
                            0.7142857142857143,
                            0.6285714285714286,
                            0.5714285714285714,
                            0.5142857142857142,
                            0.42857142857142855,
                            0.4,
                            0.37142857142857144,
                            0.3142857142857143,
                            0.2571428571428571,
                            0.17142857142857143,
                            0.08571428571428572,
                        ],
                        y=[
                            0.6410256410256411,
                            0.6111111111111112,
                            0.7407407407407407,
                            0.75,
                            0.7142857142857143,
                            0.7,
                            0.6842105263157895,
                            0.6470588235294118,
                            0.6428571428571429,
                            1.0,
                            1.0,
                        ],
                        label="tps and fps and fns",
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
    ],
)
def test__curve__plots(
    test_name: str,
    matchings: List[Union[MulticlassInferenceMatches, InferenceMatches]],
    f1_curve: CurvePlot,
    pr_curve: CurvePlot,
) -> None:
    from kolena._experimental.object_detection.utils import compute_pr_f1_plots

    f1: CurvePlot
    pr: CurvePlot
    pr, f1 = compute_pr_f1_plots(all_matches=matchings, curve_label=test_name)
    assert_curveplot_equal(f1, f1_curve)
    assert_curveplot_equal(pr, pr_curve)


@pytest.mark.metrics
@pytest.mark.parametrize(
    "plot, test_name, matchings, f1_curve, pr_curve",
    [
        (
            "all",
            "tps and fps and fns all plots",
            [
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
                            0.6756756756756757,
                            0.619718309859155,
                            0.6451612903225806,
                            0.6101694915254237,
                            0.5357142857142858,
                            0.509090909090909,
                            0.4814814814814815,
                            0.4230769230769231,
                            0.36734693877551017,
                            0.2926829268292683,
                            0.15789473684210528,
                        ],
                        label="tps and fps and fns all plots",
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
                            0.7142857142857143,
                            0.6285714285714286,
                            0.5714285714285714,
                            0.5142857142857142,
                            0.42857142857142855,
                            0.4,
                            0.37142857142857144,
                            0.3142857142857143,
                            0.2571428571428571,
                            0.17142857142857143,
                            0.08571428571428572,
                        ],
                        y=[
                            0.6410256410256411,
                            0.6111111111111112,
                            0.7407407407407407,
                            0.75,
                            0.7142857142857143,
                            0.7,
                            0.6842105263157895,
                            0.6470588235294118,
                            0.6428571428571429,
                            1.0,
                            1.0,
                        ],
                        label="tps and fps and fns all plots",
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
    ],
)
def test__curve__plots__all(
    plot: str,
    test_name: str,
    matchings: List[Union[MulticlassInferenceMatches, InferenceMatches]],
    f1_curve: CurvePlot,
    pr_curve: CurvePlot,
) -> None:
    from kolena._experimental.object_detection.utils import compute_pr_f1_plots

    f1: CurvePlot
    pr: CurvePlot
    pr, f1 = compute_pr_f1_plots(all_matches=matchings, curve_label=test_name, plot=plot)
    assert_curveplot_equal(f1, f1_curve)
    assert_curveplot_equal(pr, pr_curve)


@pytest.mark.metrics
@pytest.mark.parametrize(
    "plot, test_name, matchings, curveplot",
    [
        (
            "pr",
            "tps and fps and fns pr plot",
            [
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
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(
                        x=[
                            0.7142857142857143,
                            0.6285714285714286,
                            0.5714285714285714,
                            0.5142857142857142,
                            0.42857142857142855,
                            0.4,
                            0.37142857142857144,
                            0.3142857142857143,
                            0.2571428571428571,
                            0.17142857142857143,
                            0.08571428571428572,
                        ],
                        y=[
                            0.6410256410256411,
                            0.6111111111111112,
                            0.7407407407407407,
                            0.75,
                            0.7142857142857143,
                            0.7,
                            0.6842105263157895,
                            0.6470588235294118,
                            0.6428571428571429,
                            1.0,
                            1.0,
                        ],
                        label="tps and fps and fns pr plot",
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "f1",
            "tps and fps and fns f1 plot",
            [
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
                            0.6756756756756757,
                            0.619718309859155,
                            0.6451612903225806,
                            0.6101694915254237,
                            0.5357142857142858,
                            0.509090909090909,
                            0.4814814814814815,
                            0.4230769230769231,
                            0.36734693877551017,
                            0.2926829268292683,
                            0.15789473684210528,
                        ],
                        label="tps and fps and fns f1 plot",
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "f1",
            "tps and fps and fns f1 plot as IMs",
            [
                InferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.3)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.2)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.1)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.01)),
                    ],
                    unmatched_gt=[
                        LabeledBoundingBox((1, 1), (2, 2), "a"),
                        LabeledBoundingBox((1, 1), (2, 2), "a"),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                    ],
                ),
            ],
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
                            0.8,
                            0.9,
                            0.99,
                        ],
                        y=[
                            0.7000000000000001,
                            0.631578947368421,
                            0.5555555555555556,
                            0.47058823529411764,
                            0.375,
                            0.3636363636363636,
                            0.19999999999999998,
                        ],
                        label="tps and fps and fns f1 plot as IMs",
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
    ],
)
def test__curve__plots__one(
    plot: str,
    test_name: str,
    matchings: List[Union[MulticlassInferenceMatches, InferenceMatches]],
    curveplot: CurvePlot,
) -> None:
    from kolena._experimental.object_detection.utils import compute_pr_f1_plots

    plots = compute_pr_f1_plots(all_matches=matchings, curve_label=test_name, plot=plot)
    assert_curveplot_equal(plots[0], curveplot)


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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0),
                    ],
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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0),
                    ],
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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0),
                        ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0),
                    ],
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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0),
                        ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0),
                    ],
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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0),
                        ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0),
                    ],
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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0),
                        ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0),
                    ],
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
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0),
                        ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0),
                    ],
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
                    unmatched_inf=[ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0)],
                ),
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((7, 7), (8, 8), "b"), ScoredLabeledBoundingBox((7, 7), (8, 8), "b", 0)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)),
                    ],
                    unmatched_inf=[ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0)],
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
                    unmatched_inf=[
                        ScoredLabeledPolygon(
                            points=[(1, 1), (1, 20), (20, 20), (20, 1)],
                            label="dog",
                            score=0.9,
                        ),
                    ],
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
                        ScoredLabeledBoundingBox((10, 10), (20, 20), "dog", 0.9),
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
                        ScoredLabeledPolygon(
                            points=[(1, 1), (1, 2), (2, 2), (2, 1)],
                            label="dog",
                            score=0.9,
                        ),
                        ScoredLabeledPolygon(
                            points=[(10, 10), (10, 20), (20, 20), (20, 10)],
                            label="cat",
                            score=0.9,
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
def test__confusion__matrix(
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
        ),
    ],
)
def test__confusion__matrix__fails(
    test_name: str,
    matchings: List[MulticlassInferenceMatches],
) -> None:
    from kolena._experimental.object_detection.utils import compute_confusion_matrix_plot

    conf_mat = compute_confusion_matrix_plot(all_matches=matchings, plot_title=test_name)
    assert conf_mat is None
