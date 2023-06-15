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

from .test_plots import TEST_MATCHING
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import ScoredBoundingBox
from kolena.workflow.metrics import InferenceMatches
from kolena.workflow.metrics import MulticlassInferenceMatches


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, matchings, expected",
    [
        (
            "no confusion, one TP per label",
            TEST_MATCHING["no confusion, one TP per label"],
            0.5,
        ),
        (
            "only confusion",
            TEST_MATCHING["only confusion"],
            0.0,
        ),
        (
            "only confusion, one TP for a",
            TEST_MATCHING["only confusion, one TP for a"],
            0.9,
        ),
        (
            "only confusion, one TP for b",
            TEST_MATCHING["only confusion, one TP for b"],
            0.9,
        ),
        (
            "ones",
            TEST_MATCHING["ones"],
            0.8,
        ),
        (
            "ones, with two matchings, TPs",
            TEST_MATCHING["ones, with two matchings, TPs"],
            0.8,
        ),
        (
            "ones, with two matchings, mixed",
            TEST_MATCHING["ones, with two matchings, mixed"],
            0.6,
        ),
        (
            "two single class matchings",
            TEST_MATCHING["two single class matchings"],
            0.8,
        ),
        (
            "two single class matchings as IMs",
            TEST_MATCHING["two single class matchings as IMs"],
            0.6,
        ),
        (
            "large",
            TEST_MATCHING["large"],
            0.2,
        ),
        (
            "only tps",
            TEST_MATCHING["only tps"],
            0.01,
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
            TEST_MATCHING["tps and fps and fns"],
            0.01,
        ),
    ],
)
def test__metrics__f1__optimal(
    test_name: str,
    matchings: List[Union[MulticlassInferenceMatches, InferenceMatches]],
    expected: Union[float, Dict[str, float]],
) -> None:
    from kolena._experimental.object_detection.utils import compute_optimal_f1_threshold

    dictionary = compute_optimal_f1_threshold(matchings)
    assert expected == dictionary


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, matchings, expected",
    [
        (
            "no confusion, one TP per label",
            TEST_MATCHING["no confusion, one TP per label"],
            {"a": 0.6, "b": 0.5},
        ),
        (
            "only confusion",
            TEST_MATCHING["only confusion"],
            {"a": 0, "b": 0},
        ),
        (
            "only confusion, one TP for a",
            TEST_MATCHING["only confusion, one TP for a"],
            {"a": 0.9, "b": 0},
        ),
        (
            "only confusion, one TP for b",
            TEST_MATCHING["only confusion, one TP for b"],
            {"a": 0, "b": 0.9},
        ),
        (
            "ones",
            TEST_MATCHING["ones"],
            {"a": 0.9, "b": 0.8},
        ),
        (
            "ones, with two matchings, TPs",
            TEST_MATCHING["ones, with two matchings, TPs"],
            {"a": 0.9, "b": 0.8},
        ),
        (
            "ones, with two matchings, mixed",
            TEST_MATCHING["ones, with two matchings, mixed"],
            {"a": 0.9, "b": 0.6},
        ),
        (
            "two single class matchings",
            TEST_MATCHING["two single class matchings"],
            {"a": 0.8, "b": 0.8},
        ),
        (
            "large",
            TEST_MATCHING["large"],
            {"cat": 0.2, "cow": 0.4, "dog": 0, "fish": 0},
        ),
        (
            "only tps",
            TEST_MATCHING["only tps"],
            {"a": 0.01, "b": 0.3, "c": 0.01, "d": 0.6, "e": 0.01},
        ),
        (
            "tps and fps and fns",
            TEST_MATCHING["tps and fps and fns"],
            {"a": 0.01, "b": 0.3, "c": 0.01, "d": 0.6, "e": 0.01},
        ),
    ],
)
def test__metrics__f1__optimal__multiclass(
    test_name: str,
    matchings: List[MulticlassInferenceMatches],
    expected: Dict[str, float],
) -> None:
    from kolena._experimental.object_detection.utils import compute_optimal_f1_threshold_multiclass

    dictionary = compute_optimal_f1_threshold_multiclass(matchings)
    assert expected == dictionary


@pytest.mark.metrics
@pytest.mark.parametrize(
    "precisions, recalls, expected",
    [
        # simple AP calculations (mostly straight horizontal lines)
        ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0.1, 0.3, 0.5, 0.7, 0.9, 1], 0.5),
        ([1, 1, 1], [0.1, 0.3, 0.5], 0.5),
        ([1, 1, 1, 1, 1], [0.05, 0.3, 0.5, 0.7, 1], 1),
        ([0.75, 0.75, 0.75, 0.75, 0.75], [0, 0.3, 0.5, 0.7, 1], 0.75),
        ([0.25, 0.25, 0.25, 0.25, 0.25], [0, 0.3, 0.5, 0.7, 1], 0.25),
        ([0.25, 0.25, 0.25, 0.25, 0.25], [0.1, 0.3, 0.5, 0.7, 1], 0.25),
        ([0.5, 0.5, 0.5, 0.5, 0.5, 0.5], [0, 0.3, 0.5, 0.7, 0.9, 1], 0.5),
        ([1, 1, 1, 1, 1], [0, 0.3, 0.5, 0.7, 1], 1),
        # curve with a straight line at precision 1 and then a minor precision decrease (almost vertical)
        ([1, 1, 1, 1, 1, 1, 0.98, 0.98, 0.93], [0.008, 0.11, 0.188, 0.40, 0.5, 0.753, 0.753, 0.768, 0.768], 0.7677),
        # almost zig zag
        (
            [1, 1, 0.75, 0.92, 0.77, 0.9, 0.9, 0.86],
            [0.003, 0.009, 0.009, 0.04, 0.05, 0.24, 0.4, 0.5],
            0.44752000000000003,
        ),
        # smoother curve between precision 0.8, 1
        ([1, 1, 0.99, 0.95, 0.90, 0.85, 0.8], [0.03, 0.10, 0.356, 0.55, 0.65, 0.7, 0.74], 0.70224),
        # straight chunk + curve
        ([1, 1, 1, 0.992, 0.973, 0.9, 0.84], [0.002, 0.1, 0.33, 0.4, 0.5, 0.6, 0.63], 0.61194),
        # another curve with precision rising and falling, but defined over a small recall
        (
            [1, 1, 0.88, 0.75, 0.82, 0.8, 0.75, 0.7],
            [0.003, 0.02, 0.02, 0.03, 0.05, 0.08, 0.08, 0.11],
            0.08959999999999999,
        ),
        # zero precisions
        ([0, 0, 0, 0], [0, 0.1, 0.2, 0.3], 0),
        ([0, 0, 0, 0], [0, 0.1, 0.2, 0.3], 0),
        # some with unsorted values
        (
            [0.9, 1, 0.75, 1, 0.92, 0.77, 0.86, 0.9],
            [0.4, 0.009, 0.009, 0.003, 0.04, 0.05, 0.5, 0.24],
            0.44752000000000003,
        ),
        ([0, 0, 0, 0], [1, 0.1, 0.5, 0], 0),
        # short
        ([0.2, 0.2], [0.4, 0.6], 0.12),
        # empty
        ([], [], 0),
        # single
        ([0.5], [0.5], 0.25),
        # stairs
        ([1, 1, 0.7, 0.7, 0.5, 0.5], [0.4, 0.5, 0.52, 0.55, 0.58, 0.6], 0.56),
        (
            [0, 0.25, 0.25, 0.4, 0.4, 0.25, 0.25, 0],
            [0.4, 0.42, 0.45, 0.48, 0.50, 0.52, 0.54, 0.56],
            0.21000000000000002,
        ),
        # sawtooth
        ([0.2, 1, 0.2, 1, 0.2, 1, 0.2], [0.2, 0.23, 0.25, 0.28, 0.3, 0.31, 0.35], 0.318),
        ([0.1, 0.4, 0.1, 0.4, 0.1, 0.4, 0.1], [0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72], 0.276),
        (
            [0.2, 1, 0.2, 1, 0.2, 1, 0.2, 1, 1, 0.1, 0.4, 0.1, 0.4, 0.1, 0.4, 0.1],
            [0.2, 0.23, 0.25, 0.28, 0.3, 0.31, 0.35, 0.4, 0.45, 0.48, 0.52, 0.56, 0.6, 0.64, 0.68, 0.72],
            0.546,
        ),
    ],
)
def test__compute_ap(precisions: List[float], recalls: List[float], expected: float) -> None:
    from kolena._experimental.object_detection.utils import compute_ap

    assert pytest.approx(compute_ap(precisions, recalls), 1e-12) == expected


def test__compute_raises_exception() -> None:
    from kolena._experimental.object_detection.utils import compute_ap

    with pytest.raises(ValueError):
        compute_ap([1, 1, 1], [1, 1])
