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
from typing import List
from typing import Optional
from typing import Tuple

import numpy as np
import pytest

from kolena._experimental.classification.utils import _roc_curve
from kolena._experimental.classification.utils import compute_confusion_matrix
from kolena._experimental.classification.utils import compute_roc_curves
from kolena._experimental.classification.utils import compute_threshold_curves
from kolena._experimental.classification.utils import create_histogram
from kolena._experimental.classification.utils import get_histogram_range
from kolena._experimental.classification.utils import get_label_confidence
from kolena.workflow.annotation import Label
from kolena.workflow.annotation import ScoredLabel
from kolena.workflow.plot import ConfusionMatrix
from kolena.workflow.plot import Histogram


@pytest.mark.parametrize(
    "label, inference_labels, expected",
    [
        ("", [], 0),
        ("a", [ScoredLabel("b", 0.1)], 0),
        ("a", [ScoredLabel("a", 0.1)], 0.1),
        (
            "a",
            [
                ScoredLabel("b", 0.1),
                ScoredLabel("c", 0.2),
                ScoredLabel("d", 0.3),
                ScoredLabel("e", 0.4),
            ],
            0,
        ),
        (
            "a",
            [
                ScoredLabel("c", 0.1),
                ScoredLabel("b", 0.2),
                ScoredLabel("a", 0.3),
                ScoredLabel("d", 0.4),
            ],
            0.3,
        ),
    ],
)
def test__get_label_confidence(
    label: str,
    inference_labels: List[ScoredLabel],
    expected: float,
) -> None:
    confidence = get_label_confidence(label, inference_labels)
    assert confidence == expected


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
    if expected is not None:
        assert isinstance(range[2], int)  # making sure the # of bin is integer


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


def test__roc_curve() -> None:
    y_true = [1, 1, 0, 1, 0, 0, 1, 0, 1]
    y_score = [0.3, 0.8, 0.5, 0.7, 0.1, 0.4, 0.9, 0.2, 0.6]
    fpr, tpr = _roc_curve(y_true, y_score)
    assert len(fpr) == len(tpr)
    assert np.allclose(fpr, [0.0, 0.0, 0.0, 0.5, 0.5, 1.0])
    assert np.allclose(tpr, [0.0, 0.2, 0.8, 0.8, 1.0, 1.0])


def test__roc_curve__invalid() -> None:
    # no negative samples
    y_true = [1, 1, 1]
    y_score = [0.1, 0.5, 0.6]
    fpr, tpr = _roc_curve(y_true, y_score)
    assert np.allclose(fpr, [])
    assert np.allclose(tpr, [0.0, 1 / 3, 1.0])

    # no positive samples
    y_true = [0, 0, 0]
    y_score = [0.1, 0.5, 0.6]
    fpr, tpr = _roc_curve(y_true, y_score)
    assert np.allclose(fpr, [0.0, 1 / 3, 1.0])
    assert np.allclose(tpr, [])


@pytest.mark.parametrize(
    "gts, infs, labels",
    [
        ([], [], None),
        ([None, None], [], None),
        (
            [
                Label(label="a"),
                Label(label="b"),
            ],
            [
                [
                    ScoredLabel(label="a", score=1.0),
                    ScoredLabel(label="b", score=0.0),
                ],
                [
                    ScoredLabel(label="a", score=0.0),
                    ScoredLabel(label="b", score=1.0),
                ],
            ],
            None,
        ),
        (
            [
                Label(label="a"),
                Label(label="b"),
            ],
            [
                [
                    ScoredLabel(label="a", score=1.0),
                    ScoredLabel(label="b", score=0.0),
                ],
                [
                    ScoredLabel(label="a", score=0.0),
                    ScoredLabel(label="b", score=1.0),
                ],
            ],
            ["c", "d"],
        ),
        (
            [
                Label(label="a"),
                Label(label="a"),
                Label(label="a"),
            ],
            [
                [ScoredLabel(label="a", score=1.0)],
                [ScoredLabel(label="a", score=0.0)],
                [ScoredLabel(label="a", score=0.0)],
            ],
            None,
        ),
        (
            [
                None,
                None,
                None,
            ],
            [
                [ScoredLabel(label="a", score=1.0)],
                [ScoredLabel(label="a", score=0.0)],
                [ScoredLabel(label="a", score=0.0)],
            ],
            None,
        ),
    ],
)
def test__compute_roc_curves__invalid(
    gts: List[Optional[Label]],
    infs: List[List[ScoredLabel]],
    labels: Optional[List[str]],
) -> None:
    assert compute_roc_curves(gts, infs, labels=labels) is None


def test__compute_roc_curves__binary() -> None:
    ground_truths = [Label("1") if gt else None for gt in [1, 1, 0, 1, 0, 0, 1, 0, 1]]

    inferences = [[ScoredLabel(label="1", score=score)] for score in [0.3, 0.8, 0.5, 0.7, 0.1, 0.4, 0.9, 0.2, 0.6]]

    roc_curves = compute_roc_curves(ground_truths, inferences)
    assert roc_curves.title == "Receiver Operating Characteristic"
    assert roc_curves.x_label == "False Positive Rate (FPR)"
    assert roc_curves.y_label == "True Positive Rate (TPR)"
    assert len(roc_curves.curves) == 1
    assert roc_curves.curves[0].label == "1"
    assert roc_curves.curves[0].x == [0.0, 0.0, 0.0, 0.5, 0.5, 1.0]
    assert roc_curves.curves[0].y == [0.0, 0.2, 0.8, 0.8, 1.0, 1.0]


def test__compute_roc_curves__multiclass() -> None:
    ground_truths = [
        Label("a"),
        Label("b"),
        Label("b"),
    ]

    inferences = [
        [ScoredLabel(label="a", score=0.9), ScoredLabel(label="b", score=0.1)],
        [ScoredLabel(label="a", score=0.5), ScoredLabel(label="b", score=0.5)],
        [ScoredLabel(label="a", score=0.4), ScoredLabel(label="b", score=0.6)],
    ]

    roc_curves = compute_roc_curves(ground_truths, inferences)
    assert roc_curves.title == "Receiver Operating Characteristic (One-vs-Rest)"
    assert roc_curves.x_label == "False Positive Rate (FPR)"
    assert roc_curves.y_label == "True Positive Rate (TPR)"
    assert len(roc_curves.curves) == 2
    assert roc_curves.curves[0].label == "a"
    assert roc_curves.curves[0].x == [0, 0, 1]
    assert roc_curves.curves[0].y == [0, 1, 1]
    assert roc_curves.curves[1].label == "b"
    assert roc_curves.curves[1].x == [0, 0, 0, 1]
    assert roc_curves.curves[1].y == [0, 0.5, 1, 1]


@pytest.mark.parametrize(
    "gts, infs",
    [
        ([], []),
        ([Label("A")], []),
        (
            [Label("A"), Label("B")],
            [ScoredLabel(label="B", score=0.0)],
        ),
        (
            [Label("A"), Label("B")],
            [
                ScoredLabel(label="B", score=0.0),
                ScoredLabel(label="A", score=1.0),
            ],
        ),
    ],
)
def test__compute_threshold_curves__invalid(
    gts: List[Optional[Label]],
    infs: List[ScoredLabel],
) -> None:
    assert compute_threshold_curves(gts, infs) is None


@pytest.mark.parametrize(
    "gts, infs, thresholds, precisions, recalls, f1s",
    [
        (
            [Label("A")],
            [ScoredLabel(label="A", score=0.0)],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            [Label("A")],
            [ScoredLabel(label="A", score=1.0)],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        ),
        (
            [
                Label("A"),
                Label("A"),
                Label("A"),
                Label("A"),
            ],
            [
                ScoredLabel(label="A", score=0.2),
                ScoredLabel(label="A", score=0.55),
                ScoredLabel(label="A", score=0.6),
                ScoredLabel(label="A", score=0.8),
            ],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0],
            [1, 1, 1, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0, 0],
            [1, 1, 1, 0.75, 0.75, 0.75, 0.75, 0.5, 0.25, 0.25, 0, 0],
            [1, 1, 1, 6 / 7, 6 / 7, 6 / 7, 6 / 7, 2 / 3, 0.4, 0.4, 0, 0],
        ),
        (
            [
                Label("A"),
                Label("A"),
                Label("A"),
                Label("A"),
            ],
            [
                ScoredLabel(label="B", score=0.2),
                ScoredLabel(label="B", score=0.55),
                ScoredLabel(label="B", score=0.6),
                ScoredLabel(label="B", score=0.8),
            ],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        ),
        (
            [
                Label("A"),
                Label("A"),
                Label("B"),
                Label("B"),
            ],
            [
                ScoredLabel(label="B", score=0.2),
                ScoredLabel(label="B", score=0.55),
                ScoredLabel(label="B", score=0.6),
                ScoredLabel(label="B", score=0.8),
            ],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.55, 0.6, 0.7, 0.8, 0.9, 1.0],
            [1 / 2, 1 / 2, 1 / 2, 2 / 3, 2 / 3, 2 / 3, 2 / 3, 1, 1, 1, 0, 0],
            [1, 1, 1, 1, 1, 1, 1, 1, 1 / 2, 1 / 2, 0, 0],
            [2 / 3, 2 / 3, 2 / 3, 4 / 5, 4 / 5, 4 / 5, 4 / 5, 1.0, 2 / 3, 2 / 3, 0, 0],
        ),
    ],
)
def test__compute_threshold_curves(
    gts: List[Optional[Label]],
    infs: List[ScoredLabel],
    thresholds: List[float],
    precisions: List[float],
    recalls: List[float],
    f1s: List[float],
) -> None:
    tol = 1e-9
    curves = compute_threshold_curves(gts, infs)

    assert len(curves) == 3
    assert pytest.approx(curves[0].x, tol) == thresholds
    assert pytest.approx(curves[0].y, tol) == precisions
    assert pytest.approx(curves[1].x, tol) == thresholds
    assert pytest.approx(curves[1].y, tol) == recalls
    assert pytest.approx(curves[2].x, tol) == thresholds
    assert pytest.approx(curves[2].y, tol) == f1s


@pytest.mark.parametrize(
    "gts, infs, thresholds",
    [
        (
            [Label("A")],
            [ScoredLabel(label="A", score=0.0)],
            [0, 1.0],
        ),
        (
            [Label("A")],
            [ScoredLabel(label="A", score=1.0)],
            [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        ),
        (
            [Label("A")],
            [ScoredLabel(label="A", score=1.0)],
            [0, 1.0, 0.8, 0.2],
        ),
    ],
)
def test__compute_threshold_curves__with_thresholds(
    gts: List[Optional[Label]],
    infs: List[ScoredLabel],
    thresholds: List[float],
) -> None:
    tol = 1e-9
    curves = compute_threshold_curves(gts, infs, thresholds=thresholds)

    assert len(curves) == 3
    assert pytest.approx(curves[0].x, tol) == sorted(thresholds)
    assert pytest.approx(curves[1].x, tol) == sorted(thresholds)
    assert pytest.approx(curves[2].x, tol) == sorted(thresholds)
