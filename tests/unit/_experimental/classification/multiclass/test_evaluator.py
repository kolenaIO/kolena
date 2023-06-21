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

from kolena.classification.multiclass import AggregateMetrics
from kolena.classification.multiclass import GroundTruth
from kolena.classification.multiclass import Inference
from kolena.classification.multiclass import PerClassMetrics
from kolena.classification.multiclass import PerImageMetrics
from kolena.classification.multiclass import TestSuiteMetrics
from kolena.classification.multiclass import ThresholdConfiguration
from kolena.classification.multiclass.evaluator import _compute_per_class_metrics
from kolena.classification.multiclass.evaluator import _compute_per_image_metrics
from kolena.classification.multiclass.evaluator import _compute_test_suite_metrics
from kolena.workflow.annotation import ClassificationLabel
from kolena.workflow.annotation import ScoredClassificationLabel


def gt(label: str) -> GroundTruth:
    return GroundTruth(classification=ClassificationLabel(label=label))


def pred(label: str, score: float) -> ScoredClassificationLabel:
    return ScoredClassificationLabel(label=label, score=score)


@pytest.mark.parametrize(
    "ground_truth, inference, threshold_configuration, expected",
    [
        (
            gt("A"),
            Inference(inferences=[pred("A", 0.3), pred("B", 0.4), pred("C", 0.3)]),
            ThresholdConfiguration(),
            PerImageMetrics(classification=pred("B", 0.4), margin=0.1, is_correct=False),
        ),
        # confidence range does not need to be [0,1]
        (
            gt("A"),
            Inference(inferences=[pred("A", 500), pred("B", 200), pred("C", 300)]),
            ThresholdConfiguration(),
            PerImageMetrics(classification=pred("A", 500), margin=200, is_correct=True),
        ),
        # threshold is applied properly
        (
            gt("A"),
            Inference(inferences=[pred("A", 0.3), pred("B", 0.4), pred("C", 0.3)]),
            ThresholdConfiguration(threshold=0.5),
            PerImageMetrics(classification=None, margin=None, is_correct=False),
        ),
        # empty inference inputs are fine (no threshold)
        (
            gt("A"),
            Inference(inferences=[]),
            ThresholdConfiguration(),
            PerImageMetrics(classification=None, margin=None, is_correct=False),
        ),
        # empty inference inputs are fine (with threshold)
        (
            gt("A"),
            Inference(inferences=[]),
            ThresholdConfiguration(threshold=0.5),
            PerImageMetrics(classification=None, margin=None, is_correct=False),
        ),
    ],
)
def test__compute_per_image_metrics(
    ground_truth: GroundTruth,
    inference: Inference,
    threshold_configuration: ThresholdConfiguration,
    expected: PerImageMetrics,
) -> None:
    computed = _compute_per_image_metrics(ground_truth, inference, threshold_configuration)
    assert computed.is_correct == expected.is_correct
    assert computed.margin == pytest.approx(computed.margin)
    assert (computed.classification is None and expected.classification is None) or (
        computed.classification._to_dict() == pytest.approx(expected.classification._to_dict())
    )


@pytest.mark.parametrize(
    "all_classes, ground_truths, per_image_metrics, expected",
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
    all_classes: List[str],
    ground_truths: List[GroundTruth],
    per_image_metrics: List[PerImageMetrics],
    expected: Dict[str, PerClassMetrics],
) -> None:
    computed = _compute_per_class_metrics(all_classes, ground_truths, per_image_metrics)
    for label, computed_per_class_metrics in computed.items():
        assert label in expected
        assert computed_per_class_metrics._to_dict() == pytest.approx(expected[label]._to_dict())


def _compute_aggregate_metrics(
    ground_truths: List[GroundTruth],
    metrics_test_samples: List[PerImageMetrics],
    metrics_by_label: Dict[str, PerClassMetrics],
    expected: AggregateMetrics,
) -> None:
    ...


@pytest.mark.parametrize(
    "per_image_metrics, aggregate_metrics, expected",
    [
        # empty inputs should yield all zeroes
        (
            [],
            [],
            TestSuiteMetrics(
                n_images=0,
                n_images_skipped=0,
                variance_Accuracy=0,
                variance_Precision_macro=0,
                variance_Recall_macro=0,
                variance_F1_macro=0,
                variance_FPR_macro=0,
            ),
        ),
        # single input should yield all zeroes
        (
            [PerImageMetrics(classification=pred("A", 0.5), margin=0.25, is_correct=True)],
            [
                AggregateMetrics(
                    n_correct=1,
                    n_incorrect=0,
                    Accuracy=1,
                    Precision_macro=1,
                    Recall_macro=1,
                    F1_macro=1,
                    FPR_macro=0,
                    PerClass=[PerClassMetrics(label="A", Precision=1, Recall=1, F1=1, FPR=1)],
                ),
            ],
            TestSuiteMetrics(
                n_images=1,
                n_images_skipped=0,
                variance_Accuracy=0,
                variance_Precision_macro=0,
                variance_Recall_macro=0,
                variance_F1_macro=0,
                variance_FPR_macro=0,
            ),
        ),
        # multiple test cases
        (
            [
                PerImageMetrics(classification=pred("A", 0.5), margin=0.25, is_correct=True),
                PerImageMetrics(classification=None, margin=None, is_correct=False),
            ],
            [
                AggregateMetrics(
                    n_correct=1,
                    n_incorrect=0,
                    Accuracy=1,
                    Precision_macro=1,
                    Recall_macro=1,
                    F1_macro=1,
                    FPR_macro=0,
                    PerClass=[PerClassMetrics(label="A", Precision=1, Recall=1, F1=1, FPR=0)],
                ),
                AggregateMetrics(
                    n_correct=0,
                    n_incorrect=1,
                    Accuracy=0,
                    Precision_macro=0,
                    Recall_macro=0,
                    F1_macro=0,
                    FPR_macro=0,
                    PerClass=[PerClassMetrics(label="A", Precision=0, Recall=0, F1=0, FPR=0)],
                ),
            ],
            TestSuiteMetrics(
                n_images=2,
                n_images_skipped=1,
                variance_Accuracy=0.25,
                variance_Precision_macro=0.25,
                variance_Recall_macro=0.25,
                variance_F1_macro=0.25,
                variance_FPR_macro=0,
            ),
        ),
    ],
)
def test__compute_test_suite_metrics(
    per_image_metrics: List[PerImageMetrics],
    aggregate_metrics: List[AggregateMetrics],
    expected: TestSuiteMetrics,
) -> None:
    computed = _compute_test_suite_metrics(per_image_metrics, aggregate_metrics)
    assert computed._to_dict() == pytest.approx(expected._to_dict())
