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
import os
from typing import List
from typing import Tuple

import pytest

import kolena
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredClassificationLabel
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.plot import ConfusionMatrix
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot
from kolena.workflow.plot import Plot
from tests.integration._experimental.object_detection.test_evaluator_multiclass_fixed import (
    assert_test_case_metrics_equals_expected,
)
from tests.integration._experimental.object_detection.test_evaluator_multiclass_fixed import (
    assert_test_case_plots_equals_expected,
)
from tests.integration._experimental.object_detection.test_evaluator_multiclass_fixed import TEST_CASE
from tests.integration._experimental.object_detection.test_evaluator_multiclass_fixed import TEST_DATA
from tests.integration.helper import fake_locator

kolena.initialize(os.environ["KOLENA_TOKEN"], verbose=True)

object_detection = pytest.importorskip("kolena._experimental.object_detection", reason="requires kolena[metrics] extra")
ObjectDetectionEvaluator = object_detection.ObjectDetectionEvaluator
ClassMetricsPerTestCase = object_detection.ClassMetricsPerTestCase
TestCaseMetrics = object_detection.TestCaseMetrics
TestSample = object_detection.TestSample
TestSampleMetrics = object_detection.TestSampleMetrics
ThresholdConfiguration = object_detection.ThresholdConfiguration


EXPECTED_COMPUTE_TEST_SAMPLE_METRICS: List[Tuple[TestSample, TestSampleMetrics]] = [
    (
        TestSample(locator=fake_locator(112, "OD"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.9),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "c", 0.8),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "d", 0.7),
            ],
            FP=[],
            FN=[],
            Confused=[],
            count_TP=4,
            count_FP=0,
            count_FN=0,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=False,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.7,
            thresholds=[
                ScoredClassificationLabel("a", 0.1),
                ScoredClassificationLabel("b", 0.4),
                ScoredClassificationLabel("c", 0.1),
                ScoredClassificationLabel("d", 0.7),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(113, "OD"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledBoundingBox((1.1, 1), (2.1, 2), "a", 0.9),
                ScoredLabeledBoundingBox((3.3, 3), (4.3, 4), "a", 0.8),
            ],
            FP=[
                ScoredLabeledBoundingBox((5.5, 5), (6.5, 6), "a", 0.7),
                ScoredLabeledBoundingBox((7.7, 7), (8.7, 8), "b", 1),
            ],
            FN=[
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "b"),
            ],
            Confused=[],
            count_TP=2,
            count_FP=2,
            count_FN=2,
            count_Confused=0,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.7,
            thresholds=[
                ScoredClassificationLabel("a", 0.1),
                ScoredClassificationLabel("b", 0.4),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(114, "OD"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.6),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.1),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.5),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.4),
            ],
            FP=[],
            FN=[],
            Confused=[],
            count_TP=4,
            count_FP=0,
            count_FN=0,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=False,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=0.6,
            min_confidence_above_t=0.1,
            thresholds=[
                ScoredClassificationLabel("a", 0.1),
                ScoredClassificationLabel("b", 0.4),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(115, "OD"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.8),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.9),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "b", 0.9),
            ],
            FP=[
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 1),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 1),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.9),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.8),
            ],
            FN=[],
            Confused=[],
            count_TP=4,
            count_FP=4,
            count_FN=0,
            count_Confused=0,
            has_TP=True,
            has_FP=True,
            has_FN=False,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.8,
            thresholds=[
                ScoredClassificationLabel("a", 0.1),
                ScoredClassificationLabel("b", 0.4),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(116, "OD"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.4),
            ],
            FP=[],
            FN=[
                LabeledBoundingBox((3, 3), (4, 4), "b"),
                LabeledBoundingBox((5, 5), (6, 6), "b"),
            ],
            Confused=[],
            count_TP=2,
            count_FP=0,
            count_FN=2,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=True,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.4,
            thresholds=[
                ScoredClassificationLabel("a", 0.1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(117, "OD"), metadata={}),
        TestSampleMetrics(
            TP=[ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.9)],
            FP=[ScoredLabeledBoundingBox((7, 7), (9, 9), "b", 0.9)],
            FN=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "c"),
            ],
            Confused=[],
            count_TP=1,
            count_FP=1,
            count_FN=3,
            count_Confused=0,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.9,
            thresholds=[
                ScoredClassificationLabel("a", 0.1),
                ScoredClassificationLabel("b", 0.4),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(118, "OD"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "c", 0.8),
            ],
            FP=[],
            FN=[],
            Confused=[],
            count_TP=2,
            count_FP=0,
            count_FN=0,
            count_Confused=0,
            has_TP=True,
            has_FP=False,
            has_FN=False,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.8,
            thresholds=[
                ScoredClassificationLabel("a", 0.1),
                ScoredClassificationLabel("c", 0.1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(20, "OD"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledBoundingBox((1, 1), (3, 2), "a", 1),
                ScoredLabeledBoundingBox((3, 3), (5, 4), "a", 1),
                ScoredLabeledBoundingBox((5, 5), (7, 6), "a", 0.9),
                ScoredLabeledBoundingBox((7, 7), (9, 8), "a", 0.8),
                ScoredLabeledBoundingBox((21, 21), (22, 22), "c", 0.9),
                ScoredLabeledBoundingBox((27, 27), (28, 28), "c", 0.6),
                ScoredLabeledBoundingBox((31, 31), (32, 32), "c", 0.5),
                ScoredLabeledBoundingBox((33, 33), (34, 34), "c", 0.4),
                ScoredLabeledBoundingBox((35, 35), (36, 36), "c", 0.3),
                ScoredLabeledBoundingBox((37, 37), (38, 38), "c", 0.2),
                ScoredLabeledBoundingBox((41, 41), (42, 42), "c", 0.1),
                ScoredLabeledBoundingBox((43, 43), (44, 44), "c", 0.1),
            ],
            FP=[
                ScoredLabeledBoundingBox((11, 11), (13.01, 12), "b", 0.5),
                ScoredLabeledBoundingBox((4, 4), (5, 5), "e", 0.8),
            ],
            FN=[
                LabeledBoundingBox((11, 11), (12, 12), "b"),
                LabeledBoundingBox((13, 13), (14, 14), "b"),
                LabeledBoundingBox((15, 15), (16, 16), "b"),
                LabeledBoundingBox((17, 17), (18, 18), "b"),
                LabeledBoundingBox((23, 23), (24, 24), "c"),
                LabeledBoundingBox((25, 25), (26, 26), "c"),
            ],
            Confused=[],
            count_TP=12,
            count_FP=2,
            count_FN=6,
            count_Confused=0,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            has_Confused=False,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.1,
            thresholds=[
                ScoredClassificationLabel("a", 0.1),
                ScoredClassificationLabel("b", 0.4),
                ScoredClassificationLabel("c", 0.1),
                ScoredClassificationLabel("e", 0.1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(21, "OD"), metadata={}),
        TestSampleMetrics(
            TP=[
                ScoredLabeledBoundingBox((23, 23), (24, 24), "e", 0.8),
                ScoredLabeledBoundingBox((25, 25), (26, 26), "e", 0.7),
                ScoredLabeledBoundingBox((27, 27), (28, 28), "e", 0.6),
                ScoredLabeledBoundingBox((31, 31), (32, 32), "e", 0.1),
            ],
            FP=[
                ScoredLabeledBoundingBox((21, 21), (22, 22), "b", 0.9),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "b", 0.7),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "b", 0.6),
            ],
            FN=[
                LabeledBoundingBox((21, 21), (22, 22), "e"),
                LabeledBoundingBox((33, 33), (34, 34), "e"),
                LabeledBoundingBox((35, 35), (36, 36), "e"),
                LabeledBoundingBox((37, 37), (38, 38), "e"),
                LabeledBoundingBox((41, 41), (42, 42), "e"),
                LabeledBoundingBox((43, 43), (44, 44), "e"),
            ],
            Confused=[
                ScoredLabeledBoundingBox((21, 21), (22, 22), "b", 0.9),
            ],
            count_TP=4,
            count_FP=3,
            count_FN=6,
            count_Confused=1,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            has_Confused=True,
            ignored=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.1,
            thresholds=[
                ScoredClassificationLabel("b", 0.4),
                ScoredClassificationLabel("e", 0.1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(22, "OD"), metadata={}),
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
            max_confidence_above_t=None,
            min_confidence_above_t=None,
            thresholds=[],
        ),
    ),
]


EXPECTED_COMPUTE_TEST_CASE_METRICS = TestCaseMetrics(
    PerClass=[
        ClassMetricsPerTestCase(
            Class="a",
            nImages=8,
            Threshold=0.1,
            Objects=18,
            Inferences=20,
            TP=15,
            FN=3,
            FP=5,
            Precision=0.75,
            Recall=5 / 6,
            F1=15 / 19,
            AP=0.625,
        ),
        ClassMetricsPerTestCase(
            Class="b",
            nImages=6,
            Threshold=0.4,
            Objects=12,
            Inferences=11,
            TP=5,
            FN=7,
            FP=6,
            Precision=5 / 11,
            Recall=5 / 12,
            F1=10 / 23,
            AP=53 / 264,
        ),
        ClassMetricsPerTestCase(
            Class="c",
            nImages=4,
            Threshold=0.1,
            Objects=13,
            Inferences=10,
            TP=10,
            FN=3,
            FP=0,
            Precision=1,
            Recall=10 / 13,
            F1=20 / 23,
            AP=10 / 13,
        ),
        ClassMetricsPerTestCase(
            Class="d",
            nImages=1,
            Threshold=0.7,
            Objects=1,
            Inferences=1,
            TP=1,
            FN=0,
            FP=0,
            Precision=1,
            Recall=1,
            F1=1,
            AP=1,
        ),
        ClassMetricsPerTestCase(
            Class="e",
            nImages=1,
            Threshold=0.1,
            Objects=10,
            Inferences=5,
            TP=4,
            FN=6,
            FP=1,
            Precision=8 / 10,
            Recall=4 / 10,
            F1=8 / 15,
            AP=0.32,
        ),
    ],
    Objects=54,
    Inferences=47,
    TP=35,
    FN=19,
    FP=12,
    nIgnored=0,
    macro_Precision=881 / 1100,
    macro_Recall=889 / 1300,
    macro_F1=23776 / 32775,
    mean_AP=125053 / 214500,
    micro_Precision=35 / 47,
    micro_Recall=35 / 54,
    micro_F1=70 / 101,
)


EXPECTED_COMPUTE_TEST_CASE_PLOTS: List[Plot] = [
    CurvePlot(
        title="F1-Score vs. Confidence Threshold Per Class",
        x_label="Confidence Threshold",
        y_label="F1-Score",
        curves=[
            Curve(
                x=[0.1, 0.4, 0.6, 0.7, 0.8, 0.9, 1],
                y=[15 / 19, 28 / 37, 13 / 18, 24 / 35, 12 / 17, 0.6, 0.4],
                label="a",
            ),
            Curve(
                x=[0.1, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1],
                y=[5 / 13, 5 / 12, 10 / 23, 4 / 11, 0.3, 6 / 19, 1 / 3, 0],
                label="b",
            ),
            Curve(
                x=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 0.9],
                y=[20 / 23, 16 / 21, 0.7, 12 / 19, 5 / 9, 8 / 17, 0.375, 1 / 7],
                label="c",
            ),
            Curve(x=[0.1, 0.6, 0.7, 0.8], y=[8 / 15, 3 / 7, 4 / 13, 1 / 6], label="e"),
        ],
        x_config=None,
        y_config=None,
    ),
    CurvePlot(
        title="Precision vs. Recall Per Class",
        x_label="Recall",
        y_label="Precision",
        curves=[
            Curve(
                x=[5 / 6, 7 / 9, 13 / 18, 6 / 9, 0.5, 5 / 18, 0],
                y=[0.75, 14 / 19, 13 / 18, 0.75, 0.75, 5 / 7, 5 / 7],
                label="a",
            ),
            Curve(x=[5 / 12, 1 / 3, 1 / 4, 0], y=[5 / 11, 0.4, 0.5, 0], label="b"),
            Curve(
                x=[10 / 13, 8 / 13, 7 / 13, 6 / 13, 5 / 13, 4 / 13, 3 / 13, 1 / 13, 0],
                y=[1, 1, 1, 1, 1, 1, 1, 1, 1],
                label="c",
            ),
            Curve(x=[1, 0], y=[1, 1], label="d"),
            Curve(x=[0.4, 0.3, 0.2, 0.1, 0], y=[0.8, 0.75, 6 / 9, 0.5, 0.5], label="e"),
        ],
        x_config=None,
        y_config=None,
    ),
    ConfusionMatrix(
        title="Confusion Matrix",
        labels=["a", "b", "c", "d", "e"],
        matrix=[[15, 0, 0, 0, 0], [0, 5, 0, 0, 0], [0, 0, 10, 0, 0], [0, 0, 0, 1, 0], [0, 1, 0, 0, 4]],
        x_label="Predicted",
        y_label="Actual",
    ),
]


@pytest.mark.metrics
def test__object_detection__multiclass_evaluator__f1_optimal() -> None:
    config = ThresholdConfiguration(
        threshold_strategy="F1-Optimal",
        iou_threshold=0.5,
        min_confidence_score=0.1,
    )

    eval = ObjectDetectionEvaluator(configurations=[config])

    test_sample_metrics = eval.compute_test_sample_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA,
        configuration=config,
    )

    assert len(eval.evaluator.threshold_cache) == 1
    assert "b" in eval.evaluator.threshold_cache[config.display_name()]
    assert eval.evaluator.threshold_cache[config.display_name()]["b"] == 0.4
    assert len(eval.evaluator.matchings_by_test_case) != 0
    assert len(eval.evaluator.matchings_by_test_case[config.display_name()]) != 0
    num_of_ignored = sum([1 for _, _, inf in TEST_DATA if inf.ignored])
    assert (
        len(eval.evaluator.matchings_by_test_case[config.display_name()][TEST_CASE.name])
        == len(TEST_DATA) - num_of_ignored
    )
    assert test_sample_metrics == EXPECTED_COMPUTE_TEST_SAMPLE_METRICS

    # test case metrics, which will populate the locators cache
    assert len(eval.evaluator.locators_by_test_case) == 0

    test_case_metrics = eval.compute_test_case_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA,
        metrics=[pair[1] for pair in EXPECTED_COMPUTE_TEST_SAMPLE_METRICS],
        configuration=config,
    )
    assert len(eval.evaluator.locators_by_test_case) == 1  # cache contains locators for one test case
    assert len(eval.evaluator.locators_by_test_case[TEST_CASE.name]) == len(TEST_DATA)
    assert_test_case_metrics_equals_expected(test_case_metrics, EXPECTED_COMPUTE_TEST_CASE_METRICS)

    # test case plots only use the cached values
    plots = eval.compute_test_case_plots(
        test_case=TEST_CASE,
        inferences=[],
        metrics=[],
        configuration=config,
    )
    assert_test_case_plots_equals_expected(plots, EXPECTED_COMPUTE_TEST_CASE_PLOTS)

    # test suite behaviour is consistent with fixed evaluator
