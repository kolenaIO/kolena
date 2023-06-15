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
            "two single class matchings as IMs",
            TEST_MATCHING["two single class matchings as IMs"],
            0.6,
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
            {"a": 0.01, "b": 0.3, "c": 0.01, "d": 0.6, "e": 0.01},
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
