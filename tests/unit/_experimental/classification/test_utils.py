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

import pytest

from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.plot import Histogram


@pytest.mark.parametrize(
    "test_name, label, inference_labels, expected",
    [
        ("empty", "", [], 0),
        ("one diff", "a", [ScoredClassificationLabel("b", 0.1)], 0),
        ("one same", "a", [ScoredClassificationLabel("a", 0.1)], 0.1),
        (
            "many diff",
            "a",
            [
                ScoredClassificationLabel("b", 0.1),
                ScoredClassificationLabel("c", 0.2),
                ScoredClassificationLabel("d", 0.3),
                ScoredClassificationLabel("e", 0.4),
            ],
            0,
        ),
        (
            "full",
            "a",
            [
                ScoredClassificationLabel("c", 0.1),
                ScoredClassificationLabel("b", 0.2),
                ScoredClassificationLabel("a", 0.3),
                ScoredClassificationLabel("d", 0.4),
            ],
            0.3,
        ),
    ],
)
def test__get__label__confidence(
    test_name: str,
    label: str,
    inference_labels: List[ScoredClassificationLabel],
    expected: float,
) -> None:
    from kolena._experimental.classification.utils import get_label_confidence

    confidence = get_label_confidence(label, inference_labels)
    assert confidence == expected


@pytest.mark.parametrize(
    "test_name, values, expected",
    [
        ("empty", [], None),
        ("low out", [0.1, 0.2, -0.3], None),
        ("high out", [0.1, 0.2, 1.3], None),
        ("one", [1.0], (0.98, 1.0, 1)),
        ("zero", [0.0], (0.0, 0.02, 1)),
        ("two", [0.0, 1.0], (0.0, 1.0, 50)),
        ("half", [0.5, 1.0], (0.5, 1.0, 25)),
        ("half unaffected", [0.5, 0.8, 0.9, 1.0], (0.5, 1.0, 25)),
        ("odd", [0.33, 0.67, 0.4, 0.5, 0.6], (0.32, 0.68, 18)),
    ],
)
def test__get__histogram__range(
    test_name: str,
    values: List[float],
    expected: Optional[Tuple[float, float, int]],
) -> None:
    from kolena._experimental.classification.utils import get_histogram_range

    range = get_histogram_range(values)
    assert range == expected


@pytest.mark.parametrize(
    "test_name, values, range, title, x_label, y_label, expected",
    [
        (
            "empty",
            [],
            (0.0, 1.0, 4),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 0.25, 0.5, 0.75, 1.0], frequency=[0, 0, 0, 0]),
        ),
        (
            "strings",
            [],
            (0.0, 1.0, 1),
            "a",
            "b",
            "c",
            Histogram(title="a", x_label="b", y_label="c", buckets=[0.0, 1.0], frequency=[0]),
        ),
        (
            "outside",
            [-0.1, 1.1],
            (0.0, 1.0, 1),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 1.0], frequency=[0]),
        ),
        (
            "inside",
            [0.1],
            (0.0, 1.0, 1),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 1.0], frequency=[1]),
        ),
        (
            "000",
            [10, 1.1, -0.1],
            (0.0, 1.0, 3),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 1 / 3, 2 / 3, 1.0], frequency=[0, 0, 0]),
        ),
        (
            "on the dot",
            [0.25, 0.75],
            (0.0, 1.0, 4),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 0.25, 0.5, 0.75, 1.0], frequency=[0, 1, 0, 1]),
        ),
        (
            "101",
            [0, 1, -0.1],
            (0.0, 1.0, 3),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 1 / 3, 2 / 3, 1.0], frequency=[1, 0, 1]),
        ),
        (
            "050",
            [0.5, 0.5, 0.34, 0.65, 0.42],
            (0.0, 1.0, 3),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 1 / 3, 2 / 3, 1.0], frequency=[0, 5, 0]),
        ),
    ],
)
def test__create__histogram(
    test_name: str,
    values: List[float],
    range: Tuple[float, float, int],
    title: str,
    x_label: str,
    y_label: str,
    expected: Histogram,
) -> None:
    from kolena._experimental.classification.utils import create_histogram

    histo = create_histogram(values, range, title, x_label, y_label)
    assert histo == expected
