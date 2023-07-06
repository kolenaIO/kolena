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

from kolena.workflow import ConfusionMatrix
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledPolygon
from kolena.workflow.metrics import InferenceMatches
from kolena.workflow.metrics import MulticlassInferenceMatches


TOLERANCE = 1e-8

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
        "zeros, but one match for label a",
        "zeros, but one match for label b",
        "zeros, but b is confused with a",
        "zeros, but a is confused with b",
    ],
)
def test__none__curve__plot(
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
            "no confusion, one TP per label",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[Curve(x=[0.5, 0.6], y=[0.6666666666666666, 0.4], label="no confusion, one TP per label")],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[Curve(x=[0.5, 0.25], y=[1.0, 1.0], label="no confusion, one TP per label")],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "only confusion",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[Curve(x=[0.7, 0.8], y=[0.0, 0.0], label="only confusion")],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[Curve(x=[0.0, 0.0], y=[0.0, 0.0], label="only confusion")],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "only confusion, one TP for a",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.7, 0.8, 0.9], y=[0.3333333333333333, 0.4, 0.5], label="only confusion, one TP for a"),
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
                        x=[0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                        y=[0.3333333333333333, 0.5, 1.0],
                        label="only confusion, one TP for a",
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "only confusion, one TP for b",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.7, 0.8, 0.9], y=[0.3333333333333333, 0.4, 0.5], label="only confusion, one TP for b"),
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
                        x=[0.3333333333333333, 0.3333333333333333, 0.3333333333333333],
                        y=[0.3333333333333333, 0.5, 1.0],
                        label="only confusion, one TP for b",
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
                    Curve(x=[0.6, 0.7, 0.8, 0.9], y=[0.5, 0.5714285714285715, 0.6666666666666666, 0.4], label="ones"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall",
                x_label="Recall",
                y_label="Precision",
                curves=[Curve(x=[0.5, 0.5, 0.5, 0.25], y=[0.5, 0.6666666666666666, 1.0, 1.0], label="ones")],
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
                        x=[0.6, 0.7, 0.8, 0.9],
                        y=[0.5, 0.5714285714285715, 0.6666666666666666, 0.4],
                        label="ones, with two matchings, TPs",
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
                        x=[0.5, 0.5, 0.5, 0.25],
                        y=[0.5, 0.6666666666666666, 1.0, 1.0],
                        label="ones, with two matchings, TPs",
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
                        x=[0.5, 0.6, 0.8, 0.9],
                        y=[0.5, 0.5714285714285715, 0.3333333333333333, 0.4],
                        label="ones, with two matchings, mixed",
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
                        x=[0.5, 0.5, 0.25, 0.25],
                        y=[0.5, 0.6666666666666666, 0.5, 1.0],
                        label="ones, with two matchings, mixed",
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
                        x=[0.7, 0.8, 0.9],
                        y=[0.6666666666666666, 0.7272727272727272, 0.5],
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
                        x=[0.6666666666666666, 0.6666666666666666, 0.3333333333333333],
                        y=[0.6666666666666666, 0.8, 1.0],
                        label="two single class matchings",
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
                        x=[0.5, 0.6, 0.7, 0.8, 0.9],
                        y=[0.6666666666666666, 0.7272727272727272, 0.6, 0.5, 0.2857142857142857],
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
                        x=[0.6666666666666666, 0.6666666666666666, 0.5, 0.3333333333333333, 0.16666666666666666],
                        y=[0.6666666666666666, 0.8, 0.75, 1.0, 1.0],
                        label="two single class matchings as IMs",
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
    f1_curve: CurvePlot,
    pr_curve: CurvePlot,
) -> None:
    from kolena._experimental.object_detection.utils import compute_f1_plot
    from kolena._experimental.object_detection.utils import compute_pr_plot
    from kolena._experimental.object_detection.utils import compute_pr_curve

    f1: CurvePlot = compute_f1_plot(all_matches=TEST_MATCHING[test_name], curve_label=test_name)
    pr: CurvePlot = compute_pr_plot(all_matches=TEST_MATCHING[test_name], curve_label=test_name)
    assert f1 == f1_curve
    assert pr == pr_curve
    assert pr.curves[0] == compute_pr_curve(all_matches=TEST_MATCHING[test_name], curve_label=test_name)


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, f1_curve, pr_curve",
    [
        (
            "no confusion, one TP per label",
            None,
            None,
        ),
        (
            "only confusion",
            None,
            None,
        ),
        (
            "only confusion, one TP for a",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold Per Class",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.8, 0.9], y=[0.5, 0.6666666666666666], label="a"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5, 0.5], y=[0.5, 1.0], label="a"),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "only confusion, one TP for b",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold Per Class",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.7, 0.9], y=[0.5, 0.6666666666666666], label="b"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5, 0.5], y=[0.5, 1.0], label="b"),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "ones",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold Per Class",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.7, 0.9], y=[0.5, 0.6666666666666666], label="a"),
                    Curve(x=[0.6, 0.8], y=[0.5, 0.6666666666666666], label="b"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5, 0.5], y=[0.5, 1.0], label="a"),
                    Curve(x=[0.5, 0.5], y=[0.5, 1.0], label="b"),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "ones, with two matchings, TPs",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold Per Class",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.7, 0.9], y=[0.5, 0.6666666666666666], label="a"),
                    Curve(x=[0.6, 0.8], y=[0.5, 0.6666666666666666], label="b"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5, 0.5], y=[0.5, 1.0], label="a"),
                    Curve(x=[0.5, 0.5], y=[0.5, 1.0], label="b"),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
        (
            "ones, with two matchings, mixed",
            CurvePlot(
                title="F1-Score vs. Confidence Threshold Per Class",
                x_label="Confidence Threshold",
                y_label="F1-Score",
                curves=[
                    Curve(x=[0.5, 0.9], y=[0.5, 0.6666666666666666], label="a"),
                    Curve(x=[0.6, 0.8], y=[0.5, 0.0], label="b"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.5, 0.5], y=[0.5, 1.0], label="a"),
                    Curve(x=[0.5, 0.0], y=[0.5, 0.0], label="b"),
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
                    Curve(x=[0.8, 0.9], y=[0.6666666666666666, 0.5], label="a"),
                    Curve(x=[0.7, 0.8, 0.9], y=[0.6666666666666666, 0.8, 0.5], label="b"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[0.6666666666666666, 0.3333333333333333], y=[0.6666666666666666, 1.0], label="a"),
                    Curve(
                        x=[0.6666666666666666, 0.6666666666666666, 0.3333333333333333],
                        y=[0.6666666666666666, 1.0, 1.0],
                        label="b",
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
                    Curve(x=[0.2, 0.3], y=[0.6666666666666666, 0.0], label="cat"),
                    Curve(
                        x=[0.4, 0.5, 0.75, 0.77, 0.9],
                        y=[0.6666666666666666, 0.5454545454545454, 0.6, 0.4444444444444445, 0.25],
                        label="cow",
                    ),
                    Curve(x=[0.1, 0.5, 0.8, 0.99], y=[0.0, 0.0, 0.0, 0.0], label="dog"),
                ],
                x_config=None,
                y_config=None,
            ),
            CurvePlot(
                title="Precision vs. Recall Per Class",
                x_label="Recall",
                y_label="Precision",
                curves=[
                    Curve(x=[1.0, 0.0], y=[0.5, 0.0], label="cat"),
                    Curve(
                        x=[
                            0.5714285714285714,
                            0.42857142857142855,
                            0.42857142857142855,
                            0.2857142857142857,
                            0.14285714285714285,
                        ],
                        y=[0.8, 0.75, 1.0, 1.0, 1.0],
                        label="cow",
                    ),
                    Curve(x=[0.0, 0.0, 0.0, 0.0], y=[0.0, 0.0, 0.0, 0.0], label="dog"),
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
                        y=[
                            1.0,
                            0.923076923076923,
                            0.8333333333333333,
                            0.7272727272727273,
                            0.6,
                            0.4444444444444445,
                            0.25,
                        ],
                        label="a",
                    ),
                    Curve(
                        x=[0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        y=[1.0, 0.9090909090909091, 0.8, 0.6666666666666666, 0.5, 0.2857142857142857],
                        label="b",
                    ),
                    Curve(x=[0.01, 0.1, 0.2, 0.3], y=[1.0, 0.8571428571428571, 0.6666666666666666, 0.4], label="c"),
                    Curve(
                        x=[0.6, 0.7, 0.8, 0.9, 0.99],
                        y=[1.0, 0.888888888888889, 0.7499999999999999, 0.5714285714285715, 0.33333333333333337],
                        label="d",
                    ),
                    Curve(x=[0.01, 0.9, 0.99], y=[1.0, 0.8, 0.5], label="e"),
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
                        x=[
                            1.0,
                            0.8571428571428571,
                            0.7142857142857143,
                            0.5714285714285714,
                            0.42857142857142855,
                            0.2857142857142857,
                            0.14285714285714285,
                        ],
                        y=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        label="a",
                    ),
                    Curve(
                        x=[1.0, 0.8333333333333334, 0.6666666666666666, 0.5, 0.3333333333333333, 0.16666666666666666],
                        y=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                        label="b",
                    ),
                    Curve(x=[1.0, 0.75, 0.5, 0.25], y=[1.0, 1.0, 1.0, 1.0], label="c"),
                    Curve(x=[1.0, 0.8, 0.6, 0.4, 0.2], y=[1.0, 1.0, 1.0, 1.0, 1.0], label="d"),
                    Curve(x=[1.0, 0.6666666666666666, 0.3333333333333333], y=[1.0, 1.0, 1.0], label="e"),
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
                        y=[
                            0.7000000000000001,
                            0.631578947368421,
                            0.5555555555555556,
                            0.47058823529411764,
                            0.375,
                            0.3636363636363636,
                            0.19999999999999998,
                        ],
                        label="a",
                    ),
                    Curve(
                        x=[0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                        y=[
                            0.6,
                            0.7999999999999999,
                            0.7142857142857143,
                            0.6153846153846154,
                            0.5,
                            0.36363636363636365,
                            0.2,
                        ],
                        label="b",
                    ),
                    Curve(
                        x=[0.01, 0.1, 0.2, 0.3, 0.7],
                        y=[0.6666666666666666, 0.5454545454545454, 0.4, 0.25, 0.0],
                        label="c",
                    ),
                    Curve(
                        x=[0.1, 0.6, 0.7, 0.8, 0.9, 0.99],
                        y=[
                            0.8333333333333333,
                            1.0,
                            0.888888888888889,
                            0.7499999999999999,
                            0.5714285714285715,
                            0.33333333333333337,
                        ],
                        label="d",
                    ),
                    Curve(x=[0.01, 0.9, 0.99], y=[0.6, 0.4444444444444445, 0.25], label="e"),
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
                        x=[
                            0.7777777777777778,
                            0.6666666666666666,
                            0.5555555555555556,
                            0.4444444444444444,
                            0.3333333333333333,
                            0.2222222222222222,
                            0.1111111111111111,
                        ],
                        y=[0.6363636363636364, 0.6, 0.5555555555555556, 0.5, 0.42857142857142855, 1.0, 1.0],
                        label="a",
                    ),
                    Curve(
                        x=[0.75, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
                        y=[0.5, 0.8571428571428571, 0.8333333333333334, 0.8, 0.75, 0.6666666666666666, 0.5],
                        label="b",
                    ),
                    Curve(
                        x=[0.6666666666666666, 0.5, 0.3333333333333333, 0.16666666666666666, 0.0],
                        y=[0.6666666666666666, 0.6, 0.5, 0.5, 0.0],
                        label="c",
                    ),
                    Curve(x=[1.0, 1.0, 0.8, 0.6, 0.4, 0.2], y=[0.7142857142857143, 1.0, 1.0, 1.0, 1.0, 1.0], label="d"),
                    Curve(
                        x=[0.42857142857142855, 0.2857142857142857, 0.14285714285714285],
                        y=[1.0, 1.0, 1.0],
                        label="e",
                    ),
                ],
                x_config=None,
                y_config=None,
            ),
        ),
    ],
)
def test__curve__plots__multiclass(
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
            TEST_MATCHING["zeros with unmatched gt and unmatched inf"],
        ),
        (
            "zeros with two matchings",
            TEST_MATCHING["zeros with two matchings"],
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
