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
import pytest

from kolena.metrics import f1_score
from kolena.metrics import fpr
from kolena.metrics import precision
from kolena.metrics import recall
from kolena.metrics import specificity


@pytest.mark.parametrize(
    "true_positives, false_positives, expected",
    [
        (0, 0, 0),
        (1, 0, 1),
        (10, 0, 1),
        (9, 1, 0.9),
        (0, 1, 0),
        (0, 10, 0),
        # gracefully handle float values
        (10.0, 0.0, 1),
        (5.0, 5.0, 0.5),
        (0.0, 2.0, 0),
        # gracefully handle NaN values
        (float("nan"), 10, 0),
        (10, float("nan"), 0),
    ],
)
def test__precision(true_positives: int, false_positives: int, expected: float) -> None:
    assert precision(true_positives, false_positives) == pytest.approx(expected)


# formula is the same as precision, only semantics differ
@pytest.mark.parametrize("true_positives, false_negatives, expected", test__precision.pytestmark[0].args[1])
def test__recall(true_positives: int, false_negatives: int, expected: float) -> None:
    assert recall(true_positives, false_negatives) == pytest.approx(expected)


# formula is the same as precision, only semantics differ
@pytest.mark.parametrize("true_negatives, false_positives, expected", test__precision.pytestmark[0].args[1])
def test__specificity(true_negatives: int, false_positives: int, expected: float) -> None:
    assert specificity(true_negatives, false_positives) == pytest.approx(expected)


# formula is the same as precision, only semantics differ
@pytest.mark.parametrize("false_positives, true_negatives, expected", test__precision.pytestmark[0].args[1])
def test__fpr(true_negatives: int, false_positives: int, expected: float) -> None:
    assert fpr(true_negatives, false_positives) == pytest.approx(expected)


@pytest.mark.parametrize(
    "true_positives, false_positives, false_negatives, expected",
    [
        (0, 0, 0, 0),
        (10, 0, 0, 1),
        (1, 1, 1, 0.5),
        (9, 1, 1, 0.9),
        (1, 1, 0, 2 / 3),
        (1.0, 0, 2.0, 0.5),
        (float("nan"), 1, 0, 0),
        (1, float("nan"), 0, 0),
        (1, 1, float("nan"), 0),
    ],
)
def test__f1_score(true_positives: int, false_positives: int, false_negatives: int, expected: float) -> None:
    assert f1_score(true_positives, false_positives, false_negatives) == pytest.approx(expected)
