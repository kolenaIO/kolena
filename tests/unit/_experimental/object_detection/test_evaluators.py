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
from typing import Dict
from typing import List
from typing import Set
from typing import Tuple

import pytest

from kolena.metrics import InferenceMatches
from kolena.metrics import MulticlassInferenceMatches
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredLabel
from kolena.workflow.annotation import ScoredLabeledBoundingBox

object_detection = pytest.importorskip("kolena._experimental.object_detection", reason="requires kolena[metrics] extra")
ClassMetricsPerTestCase = object_detection.ClassMetricsPerTestCase
TestCaseMetrics = object_detection.TestCaseMetrics
TestCaseMetricsSingleClass = object_detection.TestCaseMetricsSingleClass
TestSampleMetrics = object_detection.TestSampleMetrics
TestSampleMetricsSingleClass = object_detection.TestSampleMetricsSingleClass
TestSuiteMetrics = object_detection.TestSuiteMetrics


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, bbox_matches, thresholds, expected",
    [
        (
            "empty",
            MulticlassInferenceMatches(
                matched=[],
                unmatched_gt=[],
                unmatched_inf=[],
            ),
            {"a": 0.5},
            TestSampleMetrics(
                TP=[],
                FP=[],
                FN=[],
                Confused=[],
                count_TP=0,
                count_FP=0,
                count_FN=0,
                count_Confused=0,
                has_TP=False,
                has_FP=False,
                has_FN=False,
                has_Confused=False,
                ignored=False,
                max_confidence_above_t=None,
                min_confidence_above_t=None,
                thresholds=[],
            ),
        ),
        (
            "tp",
            MulticlassInferenceMatches(
                matched=[
                    (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)),
                ],
                unmatched_gt=[],
                unmatched_inf=[],
            ),
            {"a": 0.5},
            TestSampleMetrics(
                TP=[ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)],
                FP=[],
                FN=[],
                Confused=[],
                count_TP=1,
                count_FP=0,
                count_FN=0,
                count_Confused=0,
                has_TP=True,
                has_FP=False,
                has_FN=False,
                has_Confused=False,
                ignored=False,
                max_confidence_above_t=0.8,
                min_confidence_above_t=0.8,
                thresholds=[ScoredLabel("a", 0.5)],
            ),
        ),
        (
            "tp with extra",
            MulticlassInferenceMatches(
                matched=[
                    (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)),
                    (LabeledBoundingBox((2, 2), (3, 3), "a"), ScoredLabeledBoundingBox((2, 2), (3, 3), "a", 0.1)),
                ],
                unmatched_gt=[],
                unmatched_inf=[],
            ),
            {"a": 0.5},
            TestSampleMetrics(
                TP=[ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)],
                FP=[],
                FN=[LabeledBoundingBox((2, 2), (3, 3), "a")],
                Confused=[],
                count_TP=1,
                count_FP=0,
                count_FN=1,
                count_Confused=0,
                has_TP=True,
                has_FP=False,
                has_FN=True,
                has_Confused=False,
                ignored=False,
                max_confidence_above_t=0.8,
                min_confidence_above_t=0.8,
                thresholds=[ScoredLabel("a", 0.5)],
            ),
        ),
        (
            "fn",
            MulticlassInferenceMatches(
                matched=[],
                unmatched_gt=[(LabeledBoundingBox((1, 1), (2, 2), "a"), None)],
                unmatched_inf=[],
            ),
            {"a": 0.5},
            TestSampleMetrics(
                TP=[],
                FP=[],
                FN=[LabeledBoundingBox((1, 1), (2, 2), "a")],
                Confused=[],
                count_TP=0,
                count_FP=0,
                count_FN=1,
                count_Confused=0,
                has_TP=False,
                has_FP=False,
                has_FN=True,
                has_Confused=False,
                ignored=False,
                max_confidence_above_t=None,
                min_confidence_above_t=None,
                thresholds=[],
            ),
        ),
        (
            "confused multiclass",
            MulticlassInferenceMatches(
                matched=[],
                unmatched_gt=[
                    (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8)),
                    (LabeledBoundingBox((2, 2), (3, 3), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1)),
                ],
                unmatched_inf=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1),
                ],
            ),
            {"a": 0.5, "b": 0.3},
            TestSampleMetrics(
                TP=[],
                FP=[ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8)],
                FN=[LabeledBoundingBox((1, 1), (2, 2), "a"), LabeledBoundingBox((2, 2), (3, 3), "a")],
                Confused=[ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8)],
                count_TP=0,
                count_FP=1,
                count_FN=2,
                count_Confused=1,
                has_TP=False,
                has_FP=True,
                has_FN=True,
                has_Confused=True,
                ignored=False,
                max_confidence_above_t=0.8,
                min_confidence_above_t=0.8,
                thresholds=[ScoredLabel(label="b", score=0.3)],
            ),
        ),
        (
            "fp",
            MulticlassInferenceMatches(
                matched=[],
                unmatched_gt=[],
                unmatched_inf=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.49),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                ],
            ),
            {"a": 0.5},
            TestSampleMetrics(
                TP=[],
                FP=[ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)],
                FN=[],
                Confused=[],
                count_TP=0,
                count_FP=1,
                count_FN=0,
                count_Confused=0,
                has_TP=False,
                has_FP=True,
                has_FN=False,
                has_Confused=False,
                ignored=False,
                max_confidence_above_t=0.8,
                min_confidence_above_t=0.8,
                thresholds=[ScoredLabel(label="a", score=0.5)],
            ),
        ),
        (
            "tp fp fn",
            MulticlassInferenceMatches(
                matched=[
                    (LabeledBoundingBox((1, 2), (3, 3), "a"), ScoredLabeledBoundingBox((1, 2), (3, 3), "a", 0.7)),
                    (LabeledBoundingBox((2, 2), (3, 3), "a"), ScoredLabeledBoundingBox((2, 2), (3, 3), "a", 0.1)),
                ],
                unmatched_gt=[(LabeledBoundingBox((12, 2), (13, 3), "a"), None)],
                unmatched_inf=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.49),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.6),
                ],
            ),
            {"a": 0.5},
            TestSampleMetrics(
                TP=[ScoredLabeledBoundingBox((1, 2), (3, 3), "a", 0.7)],
                FP=[ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.6)],
                FN=[LabeledBoundingBox((12, 2), (13, 3), "a"), LabeledBoundingBox((2, 2), (3, 3), "a")],
                Confused=[],
                count_TP=1,
                count_FP=1,
                count_FN=2,
                count_Confused=0,
                has_TP=True,
                has_FP=True,
                has_FN=True,
                has_Confused=False,
                ignored=False,
                max_confidence_above_t=0.7,
                min_confidence_above_t=0.6,
                thresholds=[ScoredLabel("a", 0.5)],
            ),
        ),
        (
            "tp fp fn - multiclass",
            MulticlassInferenceMatches(
                matched=[
                    (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99)),
                    (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9)),
                    (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.2)),
                    (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.1)),
                    (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.6)),
                    (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.5)),
                    (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.4)),
                    (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.1)),
                    (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.01)),
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
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.7),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.9),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.1),
                ],
            ),
            {"a": 0.5, "b": 0.2, "c": 0.99, "d": 0.0},
            TestSampleMetrics(
                TP=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.6),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.5),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.4),
                ],
                FP=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.9),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.1),
                ],
                FN=[
                    LabeledBoundingBox((1, 1), (2, 2), "a"),
                    LabeledBoundingBox((1, 1), (2, 2), "a"),
                    LabeledBoundingBox((1, 1), (2, 2), "b"),
                    LabeledBoundingBox((1, 1), (2, 2), "b"),
                    LabeledBoundingBox((1, 1), (2, 2), "c"),
                    LabeledBoundingBox((1, 1), (2, 2), "c"),
                    LabeledBoundingBox((1, 1), (2, 2), "e"),
                    LabeledBoundingBox((1, 1), (2, 2), "e"),
                    LabeledBoundingBox((1, 1), (2, 2), "e"),
                    LabeledBoundingBox((1, 1), (2, 2), "e"),
                    LabeledBoundingBox((1, 1), (2, 2), "a"),
                    LabeledBoundingBox((1, 1), (2, 2), "a"),
                    LabeledBoundingBox((1, 1), (2, 2), "c"),
                    LabeledBoundingBox((1, 1), (2, 2), "c"),
                ],
                Confused=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.1),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "d", 0.1),
                ],
                count_TP=5,
                count_FP=3,
                count_FN=14,
                count_Confused=3,
                has_TP=True,
                has_FP=True,
                has_FN=True,
                has_Confused=True,
                ignored=False,
                max_confidence_above_t=0.99,
                min_confidence_above_t=0.1,
                thresholds=[
                    ScoredLabel(label="a", score=0.5),
                    ScoredLabel(label="b", score=0.2),
                    ScoredLabel(label="c", score=0.99),
                    ScoredLabel(label="d", score=0.0),
                ],
            ),
        ),
    ],
)
def test__object_detection__multiclass__test_sample_metrics(
    test_name: str,
    bbox_matches: MulticlassInferenceMatches,
    thresholds: Dict[str, float],
    expected: TestSampleMetrics,
) -> None:
    from kolena._experimental.object_detection.evaluator_multiclass import MulticlassObjectDetectionEvaluator

    od_multi = MulticlassObjectDetectionEvaluator()
    result = od_multi.test_sample_metrics(
        bbox_matches=bbox_matches,
        thresholds=thresholds,
    )
    assert expected == result


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, bbox_matches, thresholds, expected",
    [
        (
            "empty",
            InferenceMatches(
                matched=[],
                unmatched_gt=[],
                unmatched_inf=[],
            ),
            0.5,
            TestSampleMetricsSingleClass(
                TP=[],
                FP=[],
                FN=[],
                count_TP=0,
                count_FP=0,
                count_FN=0,
                has_TP=False,
                has_FP=False,
                has_FN=False,
                ignored=False,
                max_confidence_above_t=None,
                min_confidence_above_t=None,
                thresholds=0.5,
            ),
        ),
        (
            "tp",
            InferenceMatches(
                matched=[
                    (
                        LabeledBoundingBox((1, 1), (2, 2), "a"),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                    ),
                ],
                unmatched_gt=[],
                unmatched_inf=[],
            ),
            0.5,
            TestSampleMetricsSingleClass(
                TP=[ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)],
                FP=[],
                FN=[],
                count_TP=1,
                count_FP=0,
                count_FN=0,
                has_TP=True,
                has_FP=False,
                has_FN=False,
                ignored=False,
                max_confidence_above_t=0.8,
                min_confidence_above_t=0.8,
                thresholds=0.5,
            ),
        ),
        (
            "tp with extra",
            InferenceMatches(
                matched=[
                    (
                        LabeledBoundingBox((1, 1), (2, 2), "a"),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                    ),
                    (
                        LabeledBoundingBox((2, 2), (3, 3), "a"),
                        ScoredLabeledBoundingBox((2, 2), (3, 3), "a", 0.1),
                    ),
                ],
                unmatched_gt=[],
                unmatched_inf=[],
            ),
            0.5,
            TestSampleMetricsSingleClass(
                TP=[ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)],
                FP=[],
                FN=[LabeledBoundingBox((2, 2), (3, 3), "a")],
                count_TP=1,
                count_FP=0,
                count_FN=1,
                has_TP=True,
                has_FP=False,
                has_FN=True,
                ignored=False,
                max_confidence_above_t=0.8,
                min_confidence_above_t=0.8,
                thresholds=0.5,
            ),
        ),
        (
            "fn",
            InferenceMatches(
                matched=[],
                unmatched_gt=[LabeledBoundingBox((1, 1), (2, 2), "a")],
                unmatched_inf=[],
            ),
            0.5,
            TestSampleMetricsSingleClass(
                TP=[],
                FP=[],
                FN=[LabeledBoundingBox((1, 1), (2, 2), "a")],
                count_TP=0,
                count_FP=0,
                count_FN=1,
                has_TP=False,
                has_FP=False,
                has_FN=True,
                ignored=False,
                max_confidence_above_t=None,
                min_confidence_above_t=None,
                thresholds=0.5,
            ),
        ),
        (
            "fp",
            InferenceMatches(
                matched=[],
                unmatched_gt=[],
                unmatched_inf=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.49),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                ],
            ),
            0.5,
            TestSampleMetricsSingleClass(
                TP=[],
                FP=[ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)],
                FN=[],
                count_TP=0,
                count_FP=1,
                count_FN=0,
                has_TP=False,
                has_FP=True,
                has_FN=False,
                ignored=False,
                max_confidence_above_t=0.8,
                min_confidence_above_t=0.8,
                thresholds=0.5,
            ),
        ),
        (
            "tp fp fn",
            InferenceMatches(
                matched=[
                    (LabeledBoundingBox((1, 2), (3, 3), "a"), ScoredLabeledBoundingBox((1, 2), (3, 3), "a", 0.7)),
                    (LabeledBoundingBox((2, 2), (3, 3), "a"), ScoredLabeledBoundingBox((2, 2), (3, 3), "a", 0.1)),
                ],
                unmatched_gt=[LabeledBoundingBox((12, 2), (13, 3), "a")],
                unmatched_inf=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.49),
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.6),
                ],
            ),
            0.5,
            TestSampleMetricsSingleClass(
                TP=[ScoredLabeledBoundingBox((1, 2), (3, 3), "a", 0.7)],
                FP=[
                    ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.6),
                ],
                FN=[LabeledBoundingBox((12, 2), (13, 3), "a"), LabeledBoundingBox((2, 2), (3, 3), "a")],
                count_TP=1,
                count_FP=1,
                count_FN=2,
                has_TP=True,
                has_FP=True,
                has_FN=True,
                ignored=False,
                max_confidence_above_t=0.7,
                min_confidence_above_t=0.6,
                thresholds=0.5,
            ),
        ),
    ],
)
def test__object_detection__single_class__test_sample_metrics_single_class(
    test_name: str,
    bbox_matches: InferenceMatches,
    thresholds: float,
    expected: TestSampleMetricsSingleClass,
) -> None:
    from kolena._experimental.object_detection.evaluator_single_class import SingleClassObjectDetectionEvaluator

    od_single = SingleClassObjectDetectionEvaluator()
    result = od_single.test_sample_metrics_single_class(
        bbox_matches=bbox_matches,
        thresholds=thresholds,
    )
    assert expected == result


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, metrics, average_precision, expected",
    [
        (
            "empty",
            [],
            0.0,
            TestCaseMetricsSingleClass(
                Objects=0,
                Inferences=0,
                TP=0,
                FN=0,
                FP=0,
                nIgnored=0,
                Precision=0,
                Recall=0,
                F1=0,
                AP=0,
            ),
        ),
        (
            "one ignored sample",
            [
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[],
                    count_TP=0,
                    count_FP=0,
                    count_FN=0,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    ignored=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0,
                ),
            ],
            0.0,
            TestCaseMetricsSingleClass(
                Objects=0,
                Inferences=0,
                TP=0,
                FN=0,
                FP=0,
                nIgnored=1,
                Precision=0,
                Recall=0,
                F1=0,
                AP=0,
            ),
        ),
        (
            "one fake test sample",
            [
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[],
                    count_TP=0,
                    count_FP=1,
                    count_FN=0,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    ignored=False,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0,
                ),
            ],
            0.0,
            TestCaseMetricsSingleClass(
                Objects=0,
                Inferences=1,
                TP=0,
                FN=0,
                FP=1,
                nIgnored=0,
                Precision=0,
                Recall=0,
                F1=0,
                AP=0,
            ),
        ),
        (
            "one fake test sample (large)",
            [
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=49,
                    count_FN=99,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    ignored=False,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0,
                ),
            ],
            0.0002,
            TestCaseMetricsSingleClass(
                Objects=100,
                Inferences=50,
                TP=1,
                FN=99,
                FP=49,
                nIgnored=0,
                Precision=0.02,
                Recall=0.01,
                F1=0.04 / 3,
                AP=0.0002,
            ),
        ),
        (
            "one fake test sample (large), one ignored",
            [
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[],
                    count_TP=1,
                    count_FP=49,
                    count_FN=99,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    ignored=False,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0,
                ),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[],
                    count_TP=0,
                    count_FP=0,
                    count_FN=0,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    ignored=True,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0,
                ),
            ],
            0.0002,
            TestCaseMetricsSingleClass(
                Objects=100,
                Inferences=50,
                TP=1,
                FN=99,
                FP=49,
                nIgnored=1,
                Precision=0.02,
                Recall=0.01,
                F1=0.04 / 3,
                AP=0.0002,
            ),
        ),
        (
            "2 fake test samples",
            [
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[],
                    count_TP=2,
                    count_FP=4,
                    count_FN=6,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    ignored=False,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0,
                ),
                TestSampleMetricsSingleClass(
                    TP=[],
                    FP=[],
                    FN=[],
                    count_TP=0,
                    count_FP=20,
                    count_FN=40,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    ignored=False,
                    max_confidence_above_t=None,
                    min_confidence_above_t=None,
                    thresholds=0,
                ),
            ],
            0.123,
            TestCaseMetricsSingleClass(
                Objects=48,
                Inferences=26,
                TP=2,
                FN=46,
                FP=24,
                nIgnored=0,
                Precision=1 / 13,
                Recall=1 / 24,
                F1=2 / 37,
                AP=0.123,
            ),
        ),
    ],
)
def test__object_detection__single_class__test_case_metrics_single_class(
    test_name: str,
    metrics: List[TestSampleMetricsSingleClass],
    average_precision: float,
    expected: TestCaseMetricsSingleClass,
) -> None:
    from kolena._experimental.object_detection.evaluator_single_class import SingleClassObjectDetectionEvaluator

    od_single = SingleClassObjectDetectionEvaluator()
    result = od_single.test_case_metrics_single_class(
        metrics=metrics,
        average_precision=average_precision,
    )
    assert expected == result


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, matchings, label, expected",
    [
        (
            "empty",
            [],
            "a",
            (
                MulticlassInferenceMatches(
                    matched=[],
                    unmatched_gt=[],
                    unmatched_inf=[],
                ),
                0,
            ),
        ),
        (
            "one sample",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99)),
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8)),
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.7)),
                        (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.3)),
                        (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1)),
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.7)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                        (LabeledBoundingBox((1, 1), (2, 2), "e"), None),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.7),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2),
                    ],
                ),
            ],
            "a",
            (
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                    ],
                ),
                1,
            ),
        ),
        (
            "two samples but one relevant",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99)),
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8)),
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.7)),
                        (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.3)),
                        (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1)),
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.7)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                        (LabeledBoundingBox((1, 1), (2, 2), "e"), None),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.7),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2),
                    ],
                ),
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2)),
                    ],
                    unmatched_gt=[(LabeledBoundingBox((1, 1), (2, 2), "e"), None)],
                    unmatched_inf=[ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2)],
                ),
            ],
            "a",
            (
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                    ],
                ),
                1,
            ),
        ),
        (
            "two relevant",
            [
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99)),
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8)),
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.7)),
                        (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.3)),
                        (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1)),
                        (LabeledBoundingBox((1, 1), (2, 2), "b"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.7)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                        (LabeledBoundingBox((1, 1), (2, 2), "e"), None),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.7),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2),
                    ],
                ),
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "c"), ScoredLabeledBoundingBox((1, 1), (2, 2), "c", 0.2)),
                    ],
                    unmatched_gt=[(LabeledBoundingBox((1, 1), (2, 2), "e"), None)],
                    unmatched_inf=[ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.2)],
                ),
            ],
            "a",
            (
                MulticlassInferenceMatches(
                    matched=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.99)),
                    ],
                    unmatched_gt=[
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.1)),
                        (LabeledBoundingBox((1, 1), (2, 2), "a"), None),
                    ],
                    unmatched_inf=[
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8),
                        ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.2),
                    ],
                ),
                1,
            ),
        ),
    ],
)
def test__object_detection__multiclass__bbox_matches_and_count_for_one_label(
    test_name: str,
    matchings: List[MulticlassInferenceMatches],
    label: str,
    expected: Tuple[MulticlassInferenceMatches, int],
) -> None:
    from kolena._experimental.object_detection.evaluator_multiclass import MulticlassObjectDetectionEvaluator

    od_multi = MulticlassObjectDetectionEvaluator()
    result = od_multi.bbox_matches_and_count_for_one_label(
        matchings=matchings,
        label=label,
    )
    assert expected == result


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, label, thresholds, class_matches, samples_count, average_precision, expected",
    [
        (
            "empty",
            "b",
            {"a": 0.5, "b": 0.3},
            MulticlassInferenceMatches(
                matched=[],
                unmatched_gt=[],
                unmatched_inf=[],
            ),
            0,
            1,
            ClassMetricsPerTestCase(
                Class="b",
                nImages=0,
                Threshold=0.3,
                Objects=0,
                Inferences=0,
                TP=0,
                FN=0,
                FP=0,
                Precision=0,
                Recall=0,
                F1=0,
                AP=1,
            ),
        ),
        (
            "simple",
            "a",
            {"a": 0.5, "b": 0.3},
            MulticlassInferenceMatches(
                matched=[
                    (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9)),
                ],
                unmatched_gt=[
                    (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.6)),
                ],
                unmatched_inf=[
                    ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.7),
                ],
            ),
            0,
            0.25,
            ClassMetricsPerTestCase(
                Class="a",
                nImages=0,
                Threshold=0.5,
                Objects=2,
                Inferences=2,
                TP=1,
                FN=1,
                FP=1,
                Precision=0.5,
                Recall=0.5,
                F1=0.5,
                AP=0.25,
            ),
        ),
        (
            "thresholds",
            "a",
            {"a": 0.5, "b": 0.3},
            MulticlassInferenceMatches(
                matched=[
                    (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9)),
                    (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.8)),
                    (LabeledBoundingBox((1, 1), (2, 2), "a"), ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.2)),
                ],
                unmatched_gt=[
                    (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.6)),
                    (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.5)),
                    (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.4)),
                    (LabeledBoundingBox((5, 5), (6, 6), "a"), ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.3)),
                ],
                unmatched_inf=[
                    ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.7),
                    ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.1),
                ],
            ),
            0,
            4 / 21,
            ClassMetricsPerTestCase(
                Class="a",
                nImages=0,
                Threshold=0.5,
                Objects=7,
                Inferences=3,
                TP=2,
                FN=5,
                FP=1,
                Precision=2 / 3,
                Recall=2 / 7,
                F1=8 / 20,
                AP=4 / 21,
            ),
        ),
    ],
)
def test__object_detection__multiclass__class_metrics_per_test_case(
    test_name: str,
    label: str,
    thresholds: Dict[str, float],
    class_matches: MulticlassInferenceMatches,
    samples_count: int,
    average_precision: float,
    expected: ClassMetricsPerTestCase,
) -> None:
    from kolena._experimental.object_detection.evaluator_multiclass import MulticlassObjectDetectionEvaluator

    od_multi = MulticlassObjectDetectionEvaluator()
    result = od_multi.class_metrics_per_test_case(
        label=label,
        thresholds=thresholds,
        class_matches=class_matches,
        samples_count=samples_count,
        average_precision=average_precision,
    )
    assert expected == result


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, per_class_metrics, metrics, expected",
    [
        (
            "empty",
            [],
            [],
            TestCaseMetrics(
                PerClass=[],
                Objects=0,
                Inferences=0,
                TP=0,
                FN=0,
                FP=0,
                nIgnored=0,
                macro_Precision=0,
                macro_Recall=0,
                macro_F1=0,
                mean_AP=0,
                micro_Precision=0,
                micro_Recall=0,
                micro_F1=0,
            ),
        ),
        (
            "one class",
            [
                ClassMetricsPerTestCase(
                    Class="a",
                    nImages=123,
                    Threshold=0.123,
                    Objects=1,
                    Inferences=2,
                    TP=0,
                    FN=0,
                    FP=0,
                    Precision=0.0,
                    Recall=0.2,
                    F1=0.3,
                    AP=0.4,
                ),
            ],
            [],
            TestCaseMetrics(
                PerClass=[
                    ClassMetricsPerTestCase(
                        Class="a",
                        nImages=123,
                        Threshold=0.123,
                        Objects=1,
                        Inferences=2,
                        TP=0,
                        FN=0,
                        FP=0,
                        Precision=0.0,
                        Recall=0.2,
                        F1=0.3,
                        AP=0.4,
                    ),
                ],
                Objects=0,
                Inferences=0,
                TP=0,
                FN=0,
                FP=0,
                nIgnored=0,
                macro_Precision=0.0,
                macro_Recall=0.2,
                macro_F1=0.3,
                mean_AP=0.4,
                micro_Precision=0,
                micro_Recall=0,
                micro_F1=0,
            ),
        ),
        (
            "one ignored sample",
            [],
            [
                TestSampleMetrics(
                    TP=[],
                    FP=[],
                    FN=[],
                    Confused=[],
                    count_TP=0,
                    count_FP=0,
                    count_FN=0,
                    count_Confused=0,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    has_Confused=False,
                    ignored=True,
                    max_confidence_above_t=0,
                    min_confidence_above_t=0,
                    thresholds=[],
                ),
            ],
            TestCaseMetrics(
                PerClass=[],
                Objects=0,
                Inferences=0,
                TP=0,
                FN=0,
                FP=0,
                nIgnored=1,
                macro_Precision=0.0,
                macro_Recall=0.0,
                macro_F1=0.0,
                mean_AP=0.0,
                micro_Precision=0,
                micro_Recall=0,
                micro_F1=0,
            ),
        ),
        (
            "one sample",
            [],
            [
                TestSampleMetrics(
                    TP=[],
                    FP=[],
                    FN=[],
                    Confused=[],
                    count_TP=1,
                    count_FP=49,
                    count_FN=99,
                    count_Confused=5,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    has_Confused=False,
                    ignored=False,
                    max_confidence_above_t=0,
                    min_confidence_above_t=0,
                    thresholds=[],
                ),
            ],
            TestCaseMetrics(
                PerClass=[],
                Objects=100,
                Inferences=50,
                TP=1,
                FN=99,
                FP=49,
                nIgnored=0,
                macro_Precision=0.0,
                macro_Recall=0.0,
                macro_F1=0.0,
                mean_AP=0.0,
                micro_Precision=1 / 50,
                micro_Recall=1 / 100,
                micro_F1=1 / 75,
            ),
        ),
        (
            "two samples and two classes, one ignored sample",
            [
                ClassMetricsPerTestCase(
                    Class="a",
                    nImages=123,
                    Threshold=0.5,
                    Objects=0,
                    Inferences=0,
                    TP=0,
                    FN=0,
                    FP=0,
                    Precision=0.0,
                    Recall=1.0,
                    F1=0.5,
                    AP=0.0,
                ),
                ClassMetricsPerTestCase(
                    Class="b",
                    nImages=10,
                    Threshold=0.3,
                    Objects=0,
                    Inferences=0,
                    TP=0,
                    FN=0,
                    FP=0,
                    Precision=0.0,
                    Recall=1.0,
                    F1=0.5,
                    AP=1.0,
                ),
            ],
            [
                TestSampleMetrics(
                    TP=[],
                    FP=[],
                    FN=[],
                    Confused=[],
                    count_TP=1,
                    count_FP=49,
                    count_FN=99,
                    count_Confused=5,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    has_Confused=False,
                    ignored=False,
                    max_confidence_above_t=0,
                    min_confidence_above_t=0,
                    thresholds=[],
                ),
                TestSampleMetrics(
                    TP=[],
                    FP=[],
                    FN=[],
                    Confused=[],
                    count_TP=0,
                    count_FP=0,
                    count_FN=0,
                    count_Confused=0,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    has_Confused=False,
                    ignored=True,
                    max_confidence_above_t=0,
                    min_confidence_above_t=0,
                    thresholds=[],
                ),
                TestSampleMetrics(
                    TP=[],
                    FP=[],
                    FN=[],
                    Confused=[],
                    count_TP=30,
                    count_FP=30,
                    count_FN=30,
                    count_Confused=3,
                    has_TP=False,
                    has_FP=False,
                    has_FN=False,
                    has_Confused=False,
                    ignored=False,
                    max_confidence_above_t=0,
                    min_confidence_above_t=0,
                    thresholds=[],
                ),
            ],
            TestCaseMetrics(
                PerClass=[
                    ClassMetricsPerTestCase(
                        Class="a",
                        nImages=123,
                        Threshold=0.5,
                        Objects=0,
                        Inferences=0,
                        TP=0,
                        FN=0,
                        FP=0,
                        Precision=0.0,
                        Recall=1.0,
                        F1=0.5,
                        AP=0.0,
                    ),
                    ClassMetricsPerTestCase(
                        Class="b",
                        nImages=10,
                        Threshold=0.3,
                        Objects=0,
                        Inferences=0,
                        TP=0,
                        FN=0,
                        FP=0,
                        Precision=0.0,
                        Recall=1.0,
                        F1=0.5,
                        AP=1.0,
                    ),
                ],
                Objects=160,
                Inferences=110,
                TP=31,
                FN=129,
                FP=79,
                nIgnored=1,
                macro_Precision=0.0,
                macro_Recall=1.0,
                macro_F1=0.5,
                mean_AP=0.5,
                micro_Precision=31 / 110,
                micro_Recall=31 / 160,
                micro_F1=31 / 135,
            ),
        ),
    ],
)
def test__object_detection__multiclass__test_case_metrics(
    test_name: str,
    per_class_metrics: List[ClassMetricsPerTestCase],
    metrics: List[TestSampleMetrics],
    expected: TestCaseMetrics,
) -> None:
    from kolena._experimental.object_detection.evaluator_multiclass import MulticlassObjectDetectionEvaluator

    od_multi = MulticlassObjectDetectionEvaluator()
    result = od_multi.test_case_metrics(
        per_class_metrics=per_class_metrics,
        metrics=metrics,
    )
    assert expected == result


@pytest.mark.metrics
@pytest.mark.parametrize(
    "test_name, locators, aps, expected",
    [
        (
            "empty",
            {},
            [],
            TestSuiteMetrics(n_images=0, mean_AP=0.0),
        ),
        (
            "one locator",
            {"image.png"},
            [],
            TestSuiteMetrics(n_images=1, mean_AP=0.0),
        ),
        (
            "one ap",
            {},
            [0.123],
            TestSuiteMetrics(n_images=0, mean_AP=0.123),
        ),
        (
            "one each",
            {"image.png"},
            [0.123],
            TestSuiteMetrics(n_images=1, mean_AP=0.123),
        ),
        (
            "two each",
            {"image.png", "image2.png"},
            [0.1, 0.5],
            TestSuiteMetrics(n_images=2, mean_AP=0.3),
        ),
    ],
)
def test__object_detection__test_suite_metrics(
    test_name: str,
    locators: Set[str],
    aps: List[float],
    expected: TestSuiteMetrics,
) -> None:
    from kolena._experimental.object_detection.evaluator_multiclass import MulticlassObjectDetectionEvaluator
    from kolena._experimental.object_detection.evaluator_single_class import SingleClassObjectDetectionEvaluator

    od_single = SingleClassObjectDetectionEvaluator()
    od_multi = MulticlassObjectDetectionEvaluator()
    assert expected == od_multi.test_suite_metrics(
        unique_locators=locators,
        average_precisions=aps,
    )
    assert expected == od_single.test_suite_metrics(
        unique_locators=locators,
        average_precisions=aps,
    )
