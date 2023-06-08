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
from typing import Dict
from typing import List
from typing import Union

import pytest

from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import ScoredBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledPolygon
from kolena.workflow.metrics import InferenceMatches
from kolena.workflow.metrics import MulticlassInferenceMatches


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, matchings, expected",
    [
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
            {"a": 0, "b": 0},
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
            {"a": 0, "b": 0},
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
            {"a": 0, "b": 0},
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
            {"a": 0, "b": 0},
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
            {"a": 0, "b": 0},
        ),
        (
            "ones, with two matchings, TPs",
            [
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
            {"a": 0.9, "b": 0.8},
        ),
        (
            "ones, with two matchings, mixed",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.1)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.2)),
                    ],
                    unmatched_inf=[ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.2)],
                ),
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((7, 7), (8, 8), "b"), ScoredLabeledBoundingBox((7, 7), (8, 8), "b", 0.8)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((3, 3), (4, 4), "b"), ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.1)),
                    ],
                    unmatched_inf=[ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.1)],
                ),
            ],
            {"a": 0.1, "b": 0.8},
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
            {"a": 0, "b": 0},
        ),
        (
            "two single class matchings as IMs",
            [
                InferenceMatches(
                    matched=[
                        (BoundingBox((3, 3), (4, 4)), ScoredBoundingBox((3, 3), (4, 4), 0.5)),
                        (BoundingBox((6, 6), (7, 7)), ScoredBoundingBox((6, 6), (7, 7), 0.4)),
                    ],
                    unmatched_gt=[
                        BoundingBox((1, 1), (2, 2)),
                    ],
                    unmatched_inf=[
                        ScoredBoundingBox((8, 8), (9, 9), 0.4),
                    ],
                ),
                InferenceMatches(
                    matched=[
                        (BoundingBox((1, 1), (4, 4)), ScoredBoundingBox((1, 1), (4, 4), 0.9)),
                        (BoundingBox((2, 2), (7, 7)), ScoredBoundingBox((2, 2), (7, 7), 0.2)),
                    ],
                    unmatched_gt=[
                        BoundingBox((1, 1), (2, 2)),
                    ],
                    unmatched_inf=[
                        ScoredBoundingBox((8, 8), (9, 9), 0.1),
                    ],
                ),
            ],
            0.2,
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
            {"cat": 0.3, "cow": 0.4, "dog": 0.1, "fish": 0},
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
            {"a": 0.01, "b": 0.3, "c": 0.01, "d": 0.6, "e": 0.01},
        ),
        (
            "only tps as IM",
            [
                InferenceMatches(
                    matched=[
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.99)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.9)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.8)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.3)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.2)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.1)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.01)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.8)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.7)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.6)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.5)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.4)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.3)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.3)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.2)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.1)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.01)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.99)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.9)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.8)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.7)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.6)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.99)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.9)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.01)),
                    ],
                    unmatched_gt=[],
                    unmatched_inf=[],
                ),
            ],
            0.01,
        ),
        (
            "IMs",
            [
                InferenceMatches(
                    matched=[
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.99)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.9)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.8)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.8)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.7)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.6)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.5)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.99)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.9)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.8)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.7)),
                    ],
                    unmatched_gt=[],
                    unmatched_inf=[],
                ),
                InferenceMatches(
                    matched=[
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.99)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.9)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.8)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.8)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.7)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.6)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.5)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.99)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.9)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.8)),
                        (BoundingBox((1, 1), (2, 2)), ScoredBoundingBox((1, 1), (2, 2), 0.7)),
                    ],
                    unmatched_gt=[
                        BoundingBox((1, 1), (2, 2)),
                        BoundingBox((1, 1), (2, 2)),
                    ],
                    unmatched_inf=[
                        ScoredBoundingBox((1, 1), (2, 2), 0.99),
                        ScoredBoundingBox((1, 1), (2, 2), 0.9),
                        ScoredBoundingBox((1, 1), (2, 2), 0.8),
                        ScoredBoundingBox((1, 1), (2, 2), 0.7),
                    ],
                ),
            ],
            0.5,
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
            {"a": 0.01, "b": 0.3, "c": 0.7, "d": 0.6, "e": 0.01},
        ),
    ],
)
def test__metrics__f1__optimal(
    test_name: str,
    matchings: List[Union[MulticlassInferenceMatches, InferenceMatches]],
    expected: Union[float, Dict[str, float]],
) -> None:
    from kolena._experimental.object_detection.utils import compute_optimal_f1

    dictionary = compute_optimal_f1(matchings)
    assert expected == dictionary
