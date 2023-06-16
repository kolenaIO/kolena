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

import pytest

from kolena.classification.multiclass import GroundTruth
from kolena.classification.multiclass import Inference
from kolena.classification.multiclass import PerClassMetrics
from kolena.classification.multiclass import PerImageMetrics
from kolena.classification.multiclass import ThresholdConfiguration
from kolena.classification.multiclass.evaluator import _compute_per_class_metrics
from kolena.classification.multiclass.evaluator import _compute_per_image_metrics
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import ScoredClassificationLabel


def gt(label: str) -> GroundTruth:
    return GroundTruth(classification=ClassificationLabel(label=label))


def pred(label: str, score: float) -> ScoredClassificationLabel:
    return ScoredClassificationLabel(label=label, score=score)


@pytest.mark.parametrize(
    "threshold_configuration, ground_truth, inference, expected",
    [
        (
            ThresholdConfiguration(),
            gt("A"),
            Inference(inferences=[pred("A", 0.3), pred("B", 0.4), pred("C", 0.3)]),
            PerImageMetrics(classification=pred("B", 0.4), margin=0.1, is_correct=False),
        ),
        # confidence range does not need to be [0,1]
        (
            ThresholdConfiguration(),
            gt("A"),
            Inference(inferences=[pred("A", 500), pred("B", 200), pred("C", 300)]),
            PerImageMetrics(classification=pred("A", 500), margin=200, is_correct=True),
        ),
        # threshold is applied properly
        (
            ThresholdConfiguration(threshold=0.5),
            gt("A"),
            Inference(inferences=[pred("A", 0.3), pred("B", 0.4), pred("C", 0.3)]),
            PerImageMetrics(classification=None, margin=None, is_correct=False),
        ),
        # empty inference inputs are fine (no threshold)
        (
            ThresholdConfiguration(),
            gt("A"),
            Inference(inferences=[]),
            PerImageMetrics(classification=None, margin=None, is_correct=False),
        ),
        # empty inference inputs are fine (with threshold)
        (
            ThresholdConfiguration(threshold=0.5),
            gt("A"),
            Inference(inferences=[]),
            PerImageMetrics(classification=None, margin=None, is_correct=False),
        ),
    ],
)
def test__compute_per_image_metrics(
    threshold_configuration: ThresholdConfiguration,
    ground_truth: GroundTruth,
    inference: Inference,
    expected: PerImageMetrics,
) -> None:
    computed = _compute_per_image_metrics(threshold_configuration, ground_truth, inference)
    assert computed.is_correct == expected.is_correct
    assert computed.margin == pytest.approx(computed.margin)
    assert (computed.classification is None and expected.classification is None) or (
        computed.classification._to_dict() == pytest.approx(expected.classification._to_dict())
    )


@pytest.mark.parametrize(
    "all_labels, ground_truths, metrics_test_samples, expected",
    [
        # doesn't crash on empty inputs
        ([], [], [], {}),
        # all-zero output for zero samples
        (["A"], [], [], {"A": PerClassMetrics(label="A", Precision=0, Recall=0, F1=0, FPR=0)}),
        # non-empty single-class
        (
            ["A"],
            [gt("A")],
            [PerImageMetrics(classification=ScoredClassificationLabel(label="A", score=1), margin=1, is_correct=True)],
            {"A": PerClassMetrics(label="A", Precision=1, Recall=1, F1=1, FPR=0)},
        ),
        # non-empty, perfect multi-class
        (
            ["A", "B", "C"],
            [gt("A"), gt("B"), gt("C")],
            [
                PerImageMetrics(classification=pred("A", 0.5), margin=0.25, is_correct=True),
                PerImageMetrics(classification=pred("B", 0.5), margin=0.25, is_correct=True),
                PerImageMetrics(classification=pred("C", 0.5), margin=0.25, is_correct=True),
            ],
            {
                "A": PerClassMetrics(label="A", Precision=1, Recall=1, F1=1, FPR=0),
                "B": PerClassMetrics(label="B", Precision=1, Recall=1, F1=1, FPR=0),
                "C": PerClassMetrics(label="C", Precision=1, Recall=1, F1=1, FPR=0),
            },
        ),
        # non-empty, non-perfect multi-class
        (
            ["A", "B", "C"],
            [gt("A"), gt("B"), gt("C"), gt("A"), gt("A")],
            [
                PerImageMetrics(classification=pred("A", 0.5), margin=0.25, is_correct=True),
                PerImageMetrics(classification=pred("B", 0.5), margin=0.25, is_correct=True),
                PerImageMetrics(classification=pred("C", 0.5), margin=0.25, is_correct=True),
                PerImageMetrics(classification=pred("C", 0.5), margin=0.25, is_correct=False),
                PerImageMetrics(classification=pred("C", 0.5), margin=0.25, is_correct=False),
            ],
            {
                "A": PerClassMetrics(label="A", Precision=1, Recall=1 / 3, F1=(2 / 3) / (4 / 3), FPR=0),
                "B": PerClassMetrics(label="B", Precision=1, Recall=1, F1=1, FPR=0),
                "C": PerClassMetrics(label="C", Precision=1 / 3, Recall=1, F1=(2 / 3) / (4 / 3), FPR=2 / 4),
            },
        ),
        # many classes predicted, only one in test case
        (
            ["A", "B", "C"],
            [gt("A"), gt("A"), gt("A")],
            [
                PerImageMetrics(classification=pred("A", 0.5), margin=0.25, is_correct=True),
                PerImageMetrics(classification=pred("B", 0.5), margin=0.25, is_correct=False),
                PerImageMetrics(classification=pred("B", 0.5), margin=0.25, is_correct=False),
            ],
            {
                "A": PerClassMetrics(label="A", Precision=1, Recall=1 / 3, F1=(2 / 3) / (4 / 3), FPR=0),
                "B": PerClassMetrics(label="B", Precision=0, Recall=0, F1=0, FPR=2 / 3),
                "C": PerClassMetrics(label="C", Precision=0, Recall=0, F1=0, FPR=0),
            },
        ),
        # all predictions under threshold
        (
            ["A", "B", "C"],
            [gt("A"), gt("B"), gt("A")],
            [
                PerImageMetrics(classification=None, margin=None, is_correct=False),
                PerImageMetrics(classification=None, margin=None, is_correct=False),
                PerImageMetrics(classification=None, margin=None, is_correct=False),
            ],
            {
                "A": PerClassMetrics(label="A", Precision=0, Recall=0, F1=0, FPR=0),
                "B": PerClassMetrics(label="B", Precision=0, Recall=0, F1=0, FPR=0),
                "C": PerClassMetrics(label="C", Precision=0, Recall=0, F1=0, FPR=0),
            },
        ),
        # some predictions under threshold
        (
            ["A", "B", "C"],
            [gt("A"), gt("B"), gt("A")],
            [
                PerImageMetrics(classification=None, margin=None, is_correct=False),
                PerImageMetrics(classification=pred("B", 0.5), margin=0.25, is_correct=True),
                PerImageMetrics(classification=pred("C", 0.5), margin=0.25, is_correct=False),
            ],
            {
                "A": PerClassMetrics(label="A", Precision=0, Recall=0, F1=0, FPR=0),
                "B": PerClassMetrics(label="B", Precision=1, Recall=1, F1=1, FPR=0),
                "C": PerClassMetrics(label="C", Precision=0, Recall=0, F1=0, FPR=1 / 3),
            },
        ),
    ],
)
def test__compute_per_class_metrics(
    all_labels: List[str],
    ground_truths: List[GroundTruth],
    metrics_test_samples: List[PerImageMetrics],
    expected: Dict[str, PerClassMetrics],
) -> None:
    computed = _compute_per_class_metrics(all_labels, ground_truths, metrics_test_samples)
    for label, computed_per_class_metrics in computed.items():
        assert label in expected
        assert computed_per_class_metrics._to_dict() == pytest.approx(expected[label]._to_dict())
