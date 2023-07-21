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

from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.plot import BarPlot
from kolena.workflow.plot import ConfusionMatrix
from kolena.workflow.plot import Histogram

classification = pytest.importorskip("kolena._experimental.classification")
classification_utils = pytest.importorskip("kolena._experimental.classification.utils")

ClassMetricsPerTestCase = classification.ClassMetricsPerTestCase
GroundTruth = classification.GroundTruth
TestSampleMetrics = classification.TestSampleMetrics
get_label_confidence = classification_utils.get_label_confidence
get_histogram_range = classification_utils.get_histogram_range
create_histogram = classification_utils.create_histogram
compute_test_case_confidence_histograms = classification_utils.compute_test_case_confidence_histograms
metric_bar_plot_by_class = classification_utils.metric_bar_plot_by_class
compute_test_case_confusion_matrix = classification_utils.compute_test_case_confusion_matrix


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, label, inference_labels, expected",
    [
        ("empty", "", [], 0),
        ("one dif", "a", [ScoredClassificationLabel("b", 0.1)], 0),
        ("one same", "a", [ScoredClassificationLabel("a", 0.1)], 0.1),
        (
            "many dif",
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
def test__get_label_confidence(
    test_name: str,
    label: str,
    inference_labels: List[ScoredClassificationLabel],
    expected: float,
) -> None:
    confidence = get_label_confidence(label, inference_labels)
    assert confidence == expected


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, values, expected",
    [
        ("empty", [], None),
        ("low out", [0.1, 0.2, -0.3], None),
        ("high out", [0.1, 0.2, 1.3], None),
        ("one", [1.0], (0.98, 1.0, 1)),
        ("zero", [0.0], (0.0, 0.02, 1)),
        ("two", [0.0, 1.0], (0.0, 1.0, 50)),
        ("hal", [0.5, 1.0], (0.5, 1.0, 25)),
        ("half unaffected", [0.5, 0.8, 0.9, 1.0], (0.5, 1.0, 25)),
        ("odd", [0.33, 0.67, 0.4, 0.5, 0.6], (0.32, 0.68, 18)),
    ],
)
def test__get_histogram_range(
    test_name: str,
    values: List[float],
    expected: Optional[Tuple[float, float, int]],
) -> None:
    range = get_histogram_range(values)
    assert range == expected


@pytest.mark.metrics
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
def test__create_histogram(
    test_name: str,
    values: List[float],
    range: Tuple[float, float, int],
    title: str,
    x_label: str,
    y_label: str,
    expected: Histogram,
) -> None:
    histo = create_histogram(values, range, title, x_label, y_label)
    assert histo == expected


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, metrics, range, expected",
    [
        (
            "empty",
            [],
            (0.0, 1.0, 4),
            [
                Histogram(
                    title="Score Distribution (All)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[0, 0, 0, 0],
                ),
                Histogram(
                    title="Score Distribution (Correct)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[0, 0, 0, 0],
                ),
                Histogram(
                    title="Score Distribution (Incorrect)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[0, 0, 0, 0],
                ),
            ],
        ),
        (
            "an empty",
            [TestSampleMetrics(classification=None, margin=None, is_correct=False)],
            (0.0, 1.0, 4),
            [
                Histogram(
                    title="Score Distribution (All)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[0, 0, 0, 0],
                ),
                Histogram(
                    title="Score Distribution (Correct)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[0, 0, 0, 0],
                ),
                Histogram(
                    title="Score Distribution (Incorrect)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[0, 0, 0, 0],
                ),
            ],
        ),
        (
            "an incorrect",
            [TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.1), margin=0, is_correct=False)],
            (0.0, 1.0, 4),
            [
                Histogram(
                    title="Score Distribution (All)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[1, 0, 0, 0],
                ),
                Histogram(
                    title="Score Distribution (Correct)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[0, 0, 0, 0],
                ),
                Histogram(
                    title="Score Distribution (Incorrect)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[1, 0, 0, 0],
                ),
            ],
        ),
        (
            "a correct",
            [TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.1), margin=0, is_correct=True)],
            (0.0, 1.0, 4),
            [
                Histogram(
                    title="Score Distribution (All)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[1, 0, 0, 0],
                ),
                Histogram(
                    title="Score Distribution (Correct)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[1, 0, 0, 0],
                ),
                Histogram(
                    title="Score Distribution (Incorrect)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[0, 0, 0, 0],
                ),
            ],
        ),
        (
            "some correct",
            [
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.1), margin=0, is_correct=True),
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.1), margin=0, is_correct=True),
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.6), margin=0, is_correct=False),
            ],
            (0.0, 1.0, 4),
            [
                Histogram(
                    title="Score Distribution (All)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[2, 0, 1, 0],
                ),
                Histogram(
                    title="Score Distribution (Correct)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[2, 0, 0, 0],
                ),
                Histogram(
                    title="Score Distribution (Incorrect)",
                    x_label="Confidence",
                    y_label="Count",
                    buckets=[0.0, 0.25, 0.5, 0.75, 1.0],
                    frequency=[0, 0, 1, 0],
                ),
            ],
        ),
    ],
)
def test__compute_test_case_confidence_histograms(
    test_name: str,
    metrics: List[TestSampleMetrics],
    range: Tuple[float, float, int],
    expected: List[Histogram],
) -> None:
    histo = compute_test_case_confidence_histograms(metrics, range)
    assert histo == expected


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, metric_name, per_class_metrics, expected",
    [
        (
            "empty",
            "F1",
            [],
            None,
        ),
        (
            "fake metric",
            "asd",
            [
                ClassMetricsPerTestCase(
                    label="a",
                    n_correct=1,
                    n_incorrect=1,
                    Accuracy=1,
                    Precision=1,
                    Recall=1,
                    F1=1,
                    FPR=1,
                ),
            ],
            None,
        ),
        (
            "one float",
            "F1",
            [
                ClassMetricsPerTestCase(
                    label="a",
                    n_correct=1,
                    n_incorrect=1,
                    Accuracy=1,
                    Precision=1,
                    Recall=1,
                    F1=1,
                    FPR=1,
                ),
            ],
            BarPlot(title="F1 by Class", x_label="Class", y_label="F1", labels=["a"], values=[1]),
        ),
        (
            "many floats",
            "Recall",
            [
                ClassMetricsPerTestCase(
                    label="a",
                    n_correct=1,
                    n_incorrect=3,
                    Accuracy=1,
                    Precision=1,
                    Recall=0.1,
                    F1=1,
                    FPR=1,
                ),
                ClassMetricsPerTestCase(
                    label="b",
                    n_correct=1,
                    n_incorrect=2,
                    Accuracy=1,
                    Precision=1,
                    Recall=0.2,
                    F1=1,
                    FPR=1,
                ),
            ],
            BarPlot(title="Recall by Class", x_label="Class", y_label="Recall", labels=["a", "b"], values=[0.1, 0.2]),
        ),
        (
            "one int",
            "n_correct",
            [
                ClassMetricsPerTestCase(
                    label="a",
                    n_correct=1,
                    n_incorrect=1,
                    Accuracy=1,
                    Precision=1,
                    Recall=1,
                    F1=1,
                    FPR=1,
                ),
            ],
            BarPlot(title="n_correct by Class", x_label="Class", y_label="n_correct", labels=["a"], values=[1]),
        ),
        (
            "many ints",
            "n_incorrect",
            [
                ClassMetricsPerTestCase(
                    label="a",
                    n_correct=1,
                    n_incorrect=3,
                    Accuracy=1,
                    Precision=1,
                    Recall=1,
                    F1=1,
                    FPR=1,
                ),
                ClassMetricsPerTestCase(
                    label="b",
                    n_correct=1,
                    n_incorrect=2,
                    Accuracy=1,
                    Precision=1,
                    Recall=1,
                    F1=1,
                    FPR=1,
                ),
            ],
            BarPlot(
                title="n_incorrect by Class",
                x_label="Class",
                y_label="n_incorrect",
                labels=["a", "b"],
                values=[3, 2],
            ),
        ),
        (
            "filter invalids",
            "Accuracy",
            [
                ClassMetricsPerTestCase(
                    label="a",
                    n_correct=1,
                    n_incorrect=3,
                    Accuracy=1,
                    Precision=1,
                    Recall=1,
                    F1=1,
                    FPR=1,
                ),
                ClassMetricsPerTestCase(
                    label="b",
                    n_correct=1,
                    n_incorrect=2,
                    Accuracy=0,
                    Precision=1,
                    Recall=1,
                    F1=1,
                    FPR=1,
                ),
                ClassMetricsPerTestCase(
                    label="c",
                    n_correct=1,
                    n_incorrect=2,
                    Accuracy=0.5,
                    Precision=1,
                    Recall=1,
                    F1=1,
                    FPR=1,
                ),
            ],
            BarPlot(title="Accuracy by Class", x_label="Class", y_label="Accuracy", labels=["a", "c"], values=[1, 0.5]),
        ),
    ],
)
def test__metric_bar_plot_by_class(
    test_name: str,
    metric_name: str,
    per_class_metrics: List[ClassMetricsPerTestCase],
    expected: Optional[BarPlot],
) -> None:
    plot = metric_bar_plot_by_class(metric_name, per_class_metrics)
    assert plot == expected


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, ground_truths, metrics, expected",
    [
        (
            "empty",
            [],
            [],
            None,
        ),
        (
            "different length",
            [GroundTruth(classification=ClassificationLabel("a"))],
            [],
            None,
        ),
        (
            "binary - one tp",
            [GroundTruth(classification=ClassificationLabel("a"))],
            [TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=True)],
            ConfusionMatrix(title="Label Confusion Matrix", labels=["a", "Not a"], matrix=[[1, 0], [0, 0]]),
        ),
        (
            "binary - one fp",
            [GroundTruth(classification=ClassificationLabel("a"))],
            [TestSampleMetrics(classification=None, margin=None, is_correct=False)],
            ConfusionMatrix(title="Label Confusion Matrix", labels=["a", "Not a"], matrix=[[0, 1], [0, 0]]),
        ),
        (
            "binary - one fn",
            [GroundTruth(classification=None)],
            [TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=False)],
            None,
        ),
        (
            "binary - one tn",
            [GroundTruth(classification=None)],
            [TestSampleMetrics(classification=None, margin=None, is_correct=True)],
            None,
        ),
        (
            "binary - one valid fn",
            [
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=None),
            ],
            [
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=False),
            ],
            ConfusionMatrix(title="Label Confusion Matrix", labels=["a", "Not a"], matrix=[[1, 0], [1, 0]]),
        ),
        (
            "binary - one valid tn",
            [
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=None),
            ],
            [
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=None, margin=None, is_correct=True),
            ],
            ConfusionMatrix(title="Label Confusion Matrix", labels=["a", "Not a"], matrix=[[1, 0], [0, 1]]),
        ),
        (
            "binary - complete",
            [
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=None),
                GroundTruth(classification=None),
            ],
            [
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=None, margin=None, is_correct=False),
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=False),
                TestSampleMetrics(classification=None, margin=None, is_correct=True),
            ],
            ConfusionMatrix(title="Label Confusion Matrix", labels=["a", "Not a"], matrix=[[2, 1], [1, 1]]),
        ),
        (
            "multiclass - tp",
            [
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=ClassificationLabel("b")),
            ],
            [
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=ScoredClassificationLabel("b", 0.9), margin=0, is_correct=True),
            ],
            ConfusionMatrix(title="Label Confusion Matrix", labels=["a", "b"], matrix=[[1, 0], [0, 1]]),
        ),
        (
            "multiclass - fp",
            [
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=ClassificationLabel("b")),
                GroundTruth(classification=None),
            ],
            [
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=None, margin=None, is_correct=False),
                TestSampleMetrics(classification=None, margin=None, is_correct=True),
            ],
            ConfusionMatrix(
                title="Label Confusion Matrix",
                labels=["a", "b", "None"],
                matrix=[[1, 0, 0], [0, 0, 1], [0, 0, 1]],
            ),
        ),
        (
            "multiclass - fn",
            [
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=ClassificationLabel("b")),
                GroundTruth(classification=None),
            ],
            [
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=None, margin=None, is_correct=False),
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=False),
            ],
            ConfusionMatrix(
                title="Label Confusion Matrix",
                labels=["a", "b", "None"],
                matrix=[[1, 0, 0], [0, 0, 1], [1, 0, 0]],
            ),
        ),
        (
            "multiclass - tn",
            [
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=ClassificationLabel("b")),
                GroundTruth(classification=None),
            ],
            [
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=False),
                TestSampleMetrics(classification=None, margin=0, is_correct=False),
            ],
            ConfusionMatrix(
                title="Label Confusion Matrix",
                labels=["a", "b", "None"],
                matrix=[[1, 0, 0], [1, 0, 0], [0, 0, 1]],
            ),
        ),
        (
            "multiclass - complete",
            [
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=ClassificationLabel("b")),
                GroundTruth(classification=ClassificationLabel("b")),
                GroundTruth(classification=ClassificationLabel("c")),
                GroundTruth(classification=ClassificationLabel("c")),
            ],
            [
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=ScoredClassificationLabel("b", 0.9), margin=0, is_correct=False),
                TestSampleMetrics(classification=ScoredClassificationLabel("b", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=ScoredClassificationLabel("c", 0.9), margin=0, is_correct=False),
                TestSampleMetrics(classification=ScoredClassificationLabel("c", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=ScoredClassificationLabel("d", 0.9), margin=0, is_correct=False),
            ],
            ConfusionMatrix(
                title="Label Confusion Matrix",
                labels=["a", "b", "c", "d"],
                matrix=[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]],
            ),
        ),
        (
            "multiclass - complete v2",
            [
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=ClassificationLabel("a")),
                GroundTruth(classification=ClassificationLabel("b")),
                GroundTruth(classification=ClassificationLabel("b")),
                GroundTruth(classification=ClassificationLabel("c")),
                GroundTruth(classification=ClassificationLabel("c")),
            ],
            [
                TestSampleMetrics(classification=ScoredClassificationLabel("a", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=ScoredClassificationLabel("b", 0.9), margin=0, is_correct=False),
                TestSampleMetrics(classification=ScoredClassificationLabel("b", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=ScoredClassificationLabel("c", 0.9), margin=0, is_correct=False),
                TestSampleMetrics(classification=ScoredClassificationLabel("c", 0.9), margin=0, is_correct=True),
                TestSampleMetrics(classification=None, margin=None, is_correct=False),
            ],
            ConfusionMatrix(
                title="Label Confusion Matrix",
                labels=["a", "b", "c", "None"],
                matrix=[[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0]],
            ),
        ),
    ],
)
def test__compute_test_case_confusion_matrix(
    test_name: str,
    ground_truths: List[GroundTruth],
    metrics: List[TestSampleMetrics],
    expected: ConfusionMatrix,
) -> None:
    conf_mat = compute_test_case_confusion_matrix(ground_truths, metrics)
    assert conf_mat == expected
