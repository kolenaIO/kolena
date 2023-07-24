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

from kolena._experimental.classification.utils import compute_confusion_matrix
from kolena._experimental.classification.utils import create_histogram
from kolena._experimental.classification.utils import get_histogram_range
from kolena._experimental.classification.utils import get_label_confidence
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.plot import ConfusionMatrix
from kolena.workflow.plot import Histogram


@pytest.mark.metrics
@pytest.mark.parametrize(
    "label, inference_labels, expected",
    [
        ("", [], 0),
        ("a", [ScoredClassificationLabel("b", 0.1)], 0),
        ("a", [ScoredClassificationLabel("a", 0.1)], 0.1),
        (
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
def test__get_label_confidence(
    label: str,
    inference_labels: List[ScoredClassificationLabel],
    expected: float,
) -> None:
    confidence = get_label_confidence(label, inference_labels)
    assert confidence == expected


@pytest.mark.metrics
@pytest.mark.parametrize(
    "values, expected",
    [
        ([], None),
        ([0.1, 0.2, -0.3], None),
        ([0.1, 0.2, 1.3], None),
        ([1.0], (0.98, 1.0, 1)),
        ([0.0], (0.0, 0.02, 1)),
        ([0.0, 1.0], (0.0, 1.0, 50)),
        ([0.5, 1.0], (0.5, 1.0, 25)),
        ([0.5, 0.8, 0.9, 1.0], (0.5, 1.0, 25)),
        ([0.33, 0.67, 0.4, 0.5, 0.6], (0.32, 0.68, 18)),
    ],
)
def test__get_histogram_range(
    values: List[float],
    expected: Optional[Tuple[float, float, int]],
) -> None:
    range = get_histogram_range(values)
    assert range == expected


@pytest.mark.metrics
@pytest.mark.parametrize(
    "values, range, title, x_label, y_label, expected",
    [
        (
            [],
            (0.0, 1.0, 4),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 0.25, 0.5, 0.75, 1.0], frequency=[0, 0, 0, 0]),
        ),
        (
            [],
            (0.0, 1.0, 1),
            "a",
            "b",
            "c",
            Histogram(title="a", x_label="b", y_label="c", buckets=[0.0, 1.0], frequency=[0]),
        ),
        (
            [-0.1, 1.1],
            (0.0, 1.0, 1),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 1.0], frequency=[0]),
        ),
        (
            [0.1],
            (0.0, 1.0, 1),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 1.0], frequency=[1]),
        ),
        (
            [10, 1.1, -0.1],
            (0.0, 1.0, 3),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 1 / 3, 2 / 3, 1.0], frequency=[0, 0, 0]),
        ),
        (
            [0.25, 0.75],
            (0.0, 1.0, 4),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 0.25, 0.5, 0.75, 1.0], frequency=[0, 1, 0, 1]),
        ),
        (
            [0, 1, -0.1],
            (0.0, 1.0, 3),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 1 / 3, 2 / 3, 1.0], frequency=[1, 0, 1]),
        ),
        (
            [0.5, 0.5, 0.34, 0.65, 0.42],
            (0.0, 1.0, 3),
            "",
            "",
            "",
            Histogram(title="", x_label="", y_label="", buckets=[0.0, 1 / 3, 2 / 3, 1.0], frequency=[0, 5, 0]),
        ),
    ],
)
def test__create_histogram(
    values: List[float],
    range: Tuple[float, float, int],
    title: str,
    x_label: str,
    y_label: str,
    expected: Histogram,
) -> None:
    histogram = create_histogram(values, range, title, x_label, y_label)
    assert histogram == expected


@pytest.mark.metrics
@pytest.mark.parametrize(
    "gts, infs, labels, expected",
    [
        (
            [],
            [],
            None,
            None,
        ),
        (
            ["a"],
            [],
            None,
            None,
        ),
        (
            ["a"],
            ["a"],
            None,
            None,
        ),
        (
            ["a"],
            ["a"],
            ["a", "not a"],
            ConfusionMatrix(title="Confusion Matrix", labels=["a", "not a"], matrix=[[1, 0], [0, 0]]),
        ),
        (
            ["a"],
            ["not a"],
            ["a", "not a"],
            ConfusionMatrix(title="Confusion Matrix", labels=["a", "not a"], matrix=[[0, 1], [0, 0]]),
        ),
        (
            ["not a"],
            ["a"],
            ["a", "not a"],
            ConfusionMatrix(title="Confusion Matrix", labels=["a", "not a"], matrix=[[0, 0], [1, 0]]),
        ),
        (
            ["not a"],
            ["not a"],
            ["a", "not a"],
            ConfusionMatrix(title="Confusion Matrix", labels=["a", "not a"], matrix=[[0, 0], [0, 1]]),
        ),
        (
            ["a", "not a"],
            ["a", "a"],
            None,
            ConfusionMatrix(title="Confusion Matrix", labels=["a", "not a"], matrix=[[1, 0], [1, 0]]),
        ),
        (
            ["a", "not a"],
            ["a", "not a"],
            None,
            ConfusionMatrix(title="Confusion Matrix", labels=["a", "not a"], matrix=[[1, 0], [0, 1]]),
        ),
        (
            ["a", "a", "a", "not a", "not a"],
            ["a", "a", "not a", "a", "not a"],
            None,
            ConfusionMatrix(title="Confusion Matrix", labels=["a", "not a"], matrix=[[2, 1], [1, 1]]),
        ),
        (
            ["a", "b"],
            ["a", "b"],
            None,
            ConfusionMatrix(title="Confusion Matrix", labels=["a", "b"], matrix=[[1, 0], [0, 1]]),
        ),
        (
            ["a", "b"],
            ["a", "c"],
            None,
            ConfusionMatrix(
                title="Confusion Matrix",
                labels=["a", "b", "c"],
                matrix=[[1, 0, 0], [0, 0, 1], [0, 0, 0]],
            ),
        ),
        (
            ["a", "b", "b"],
            ["a", "c", "d"],
            ["a", "b", "c"],
            ConfusionMatrix(
                title="Confusion Matrix",
                labels=["a", "b", "c"],
                matrix=[[1, 0, 0], [0, 0, 1], [0, 0, 0]],
            ),
        ),
        (
            ["a", "b", "b"],
            ["a", "c", "d"],
            ["c", "b", "a"],
            ConfusionMatrix(
                title="Confusion Matrix",
                labels=["c", "b", "a"],
                matrix=[[0, 0, 0], [1, 0, 0], [0, 0, 1]],
            ),
        ),
    ],
)
def test__compute_confusion_matrix(
    gts: List[str],
    infs: List[str],
    labels: List[str],
    expected: ConfusionMatrix,
) -> None:
    conf_mat = compute_confusion_matrix(gts, infs, labels=labels)
    assert conf_mat == expected


def test__compute_confusion_matrix__with_title() -> None:
    gts = ["a", "not a"]
    infs = ["a", "a"]
    conf_mat = compute_confusion_matrix(gts, infs, title="Title Test")
    assert conf_mat == ConfusionMatrix(
        title="Title Test",
        labels=["a", "not a"],
        matrix=[[1, 0], [1, 0]],
    )
