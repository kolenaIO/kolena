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
from typing import Tuple

import pytest

from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import ScoredLabeledBoundingBox
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot
from kolena.workflow.plot import Plot
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix

object_detection = pytest.importorskip("kolena._experimental.object_detection", reason="requires kolena[metrics] extra")
GroundTruth = object_detection.GroundTruth
Inference = object_detection.Inference
TestCase = object_detection.TestCase
TestSample = object_detection.TestSample
TestSuite = object_detection.TestSuite
ThresholdConfiguration = object_detection.ThresholdConfiguration
ThresholdStrategy = object_detection.ThresholdStrategy
TestCaseMetricsSingleClass = object_detection.TestCaseMetricsSingleClass
TestSampleMetricsSingleClass = object_detection.TestSampleMetricsSingleClass
TestSuiteMetrics = object_detection.TestSuiteMetrics


TEST_CASE_NAME = "single class OD test"
TEST_CASE = TestCase(with_test_prefix(TEST_CASE_NAME + " case"))
TEST_SUITE = TestSuite(with_test_prefix(TEST_CASE_NAME + " suite"))


TEST_DATA: List[Tuple[TestSample, GroundTruth, Inference]] = [
    (
        TestSample(locator=fake_locator(112, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "a"),
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "d"),  # single class OD can have 1+ classes (not distinguished)
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.9),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.8),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "d", 0.7),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(113, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "a"),
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1.1, 1), (2.1, 2), "a", 0.9),
                ScoredLabeledBoundingBox((3.3, 3), (4.3, 4), "a", 0.8),
                ScoredLabeledBoundingBox((5.5, 5), (6.5, 6), "a", 0.7),
                ScoredLabeledBoundingBox((7.7, 7), (8.7, 8), "a", 1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(114, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "a"),
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.6),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.5),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.4),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(115, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "a"),
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 1),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 1),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.9),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.8),
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.9),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.9),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.8),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(116, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "a"),
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.4),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(117, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "a"),
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.9),
                ScoredLabeledBoundingBox((7, 7), (9, 9), "a", 0.9),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(118, "OD")),
        GroundTruth(
            bboxes=[
                LabeledBoundingBox((1, 1), (2, 2), "a"),
                LabeledBoundingBox((3, 3), (4, 4), "a"),
            ],
            ignored_bboxes=[
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
        ),
        Inference(
            bboxes=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.8),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.1),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 1),
            ],
        ),
    ),
]


EXPECTED_COMPUTE_TEST_SAMPLE_METRICS: List[Tuple[TestSample, TestSampleMetricsSingleClass]] = [
    (
        # single class OD can have 1+ classes (not distinguished)
        TestSample(locator=fake_locator(112, "OD"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.9),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.8),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "d", 0.7),
            ],
            FP=[],
            FN=[],
            count_TP=4,
            count_FP=0,
            count_FN=0,
            has_TP=True,
            has_FP=False,
            has_FN=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.7,
            thresholds=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(113, "OD"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledBoundingBox((1.1, 1), (2.1, 2), "a", 0.9),
                ScoredLabeledBoundingBox((3.3, 3), (4.3, 4), "a", 0.8),
            ],
            FP=[
                ScoredLabeledBoundingBox((7.7, 7), (8.7, 8), "a", 1),
                ScoredLabeledBoundingBox((5.5, 5), (6.5, 6), "a", 0.7),
            ],
            FN=[
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
            count_TP=2,
            count_FP=2,
            count_FN=2,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            max_confidence_above_t=1,
            min_confidence_above_t=0.7,
            thresholds=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(114, "OD"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.6),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.5),
            ],
            FP=[],
            FN=[
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
            count_TP=2,
            count_FP=0,
            count_FN=2,
            has_TP=True,
            has_FP=False,
            has_FN=True,
            max_confidence_above_t=0.6,
            min_confidence_above_t=0.5,
            thresholds=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(115, "OD"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.9),
                ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.9),
                ScoredLabeledBoundingBox((7, 7), (8, 8), "a", 0.8),
            ],
            FP=[
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 1),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 1),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.9),
                ScoredLabeledBoundingBox((0, 0), (1, 1), "a", 0.8),
            ],
            FN=[],
            count_TP=4,
            count_FP=4,
            count_FN=0,
            has_TP=True,
            has_FP=True,
            has_FN=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.8,
            thresholds=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(116, "OD"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 1),
            ],
            FP=[],
            FN=[
                LabeledBoundingBox((3, 3), (4, 4), "a"),
                LabeledBoundingBox((5, 5), (6, 6), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
            ],
            count_TP=1,
            count_FP=0,
            count_FN=3,
            has_TP=True,
            has_FP=False,
            has_FN=True,
            max_confidence_above_t=1,
            min_confidence_above_t=1,
            thresholds=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(117, "OD"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledBoundingBox((5, 5), (6, 6), "a", 0.9),
            ],
            FP=[
                ScoredLabeledBoundingBox((7, 7), (9, 9), "a", 0.9),
            ],
            FN=[
                LabeledBoundingBox((3, 3), (4, 4), "a"),
                LabeledBoundingBox((7, 7), (8, 8), "a"),
                LabeledBoundingBox((1, 1), (2, 2), "a"),
            ],
            count_TP=1,
            count_FP=1,
            count_FN=3,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.9,
            thresholds=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(118, "OD"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledBoundingBox((1, 1), (2, 2), "a", 0.9),
                ScoredLabeledBoundingBox((3, 3), (4, 4), "a", 0.8),
            ],
            FP=[],
            FN=[],
            count_TP=2,
            count_FP=0,
            count_FN=0,
            has_TP=True,
            has_FP=False,
            has_FN=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.8,
            thresholds=0.5,
        ),
    ),
]


EXPECTED_COMPUTE_TEST_CASE_METRICS = TestCaseMetricsSingleClass(
    Objects=26,
    Inferences=23,
    TP=16,
    FN=10,
    FP=7,
    Precision=16 / 23,
    Recall=16 / 26,
    F1=32 / 49,
    AP=200 / 351,
)

EXPECTED_COMPUTE_TEST_CASE_PLOTS: List[Plot] = [
    CurvePlot(
        title="Precision vs. Recall",
        x_label="Recall",
        y_label="Precision",
        curves=[
            Curve(
                label=None,
                x=[10 / 13, 19 / 26, 9 / 13, 8 / 13, 15 / 26, 7 / 13, 1 / 2, 9 / 26, 3 / 26, 0],
                y=[20 / 27, 19 / 26, 18 / 25, 16 / 23, 15 / 22, 2 / 3, 13 / 19, 9 / 14, 0.5, 0.5],
            ),
        ],
        x_config=None,
        y_config=None,
    ),
    CurvePlot(
        title="F1-Score vs. Confidence Threshold",
        x_label="Confidence Threshold",
        y_label="F1-Score",
        curves=[
            Curve(
                label=None,
                x=[0, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                y=[40 / 53, 19 / 26, 12 / 17, 32 / 49, 5 / 8, 28 / 47, 26 / 45, 9 / 20, 3 / 16],
            ),
        ],
        x_config=None,
        y_config=None,
    ),
]


def assert_test_case_metrics_equals_expected(
    metrics: TestCaseMetricsSingleClass,
    other_metrics: TestCaseMetricsSingleClass,
) -> None:
    assert metrics.Objects == other_metrics.Objects
    assert metrics.Inferences == other_metrics.Inferences
    assert metrics.TP == other_metrics.TP
    assert metrics.FN == other_metrics.FN
    assert metrics.FP == other_metrics.FP
    assert pytest.approx(metrics.Precision, abs=1e-12) == other_metrics.Precision
    assert pytest.approx(metrics.Recall, abs=1e-12) == other_metrics.Recall
    assert pytest.approx(metrics.F1, abs=1e-12) == other_metrics.F1
    assert pytest.approx(metrics.AP, abs=1e-12) == other_metrics.AP


def assert_curve(
    curve: Curve,
    expectation: Curve,
) -> None:
    assert curve.label is None
    assert expectation.label is None
    assert len(curve.x) == len(expectation.x)
    assert sum(abs(a - b) for a, b in zip(curve.x, expectation.x)) < 1e-12
    assert len(curve.y) == len(expectation.y)
    assert sum(abs(a - b) for a, b in zip(curve.y, expectation.y)) < 1e-12


def assert_test_case_plots_equals_expected(
    plots: List[Plot],
    other_plots: List[Plot],
) -> None:
    assert len(plots) == len(other_plots)
    # check curve plots
    for plot, expected in zip(plots, other_plots):
        assert plot.title == expected.title
        assert plot.x_label == expected.x_label
        assert plot.y_label == expected.y_label
        assert len(plot.curves) == 1
        assert len(expected.curves) == 1
        assert_curve(plot.curves[0], expected.curves[0])
        assert plot.x_config == expected.x_config
        assert plot.y_config == expected.y_config


@pytest.mark.metrics
def test__object_detection__multiclass_evaluator__fixed() -> None:
    from kolena._experimental.object_detection import ObjectDetectionEvaluator

    config = ThresholdConfiguration(
        threshold_strategy=ThresholdStrategy.FIXED_05,
        iou_threshold=0.5,
        min_confidence_score=0,
        with_class_level_metrics=False,
    )
    eval = ObjectDetectionEvaluator(configurations=[config])

    test_sample_metrics = eval.compute_test_sample_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA,
        configuration=config,
    )
    assert len(eval.evaluator.threshold_cache) == 0  # empty because not f1 optimal config
    assert len(eval.evaluator.matchings_by_test_case) != 0
    assert len(eval.evaluator.matchings_by_test_case[TEST_CASE.name]) == len(TEST_DATA)
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

    # test suite metrics - one
    test_suite_metrics = eval.compute_test_suite_metrics(
        test_suite=TEST_SUITE,
        metrics=[
            (TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS),
        ],
        configuration=config,
    )
    assert test_suite_metrics == TestSuiteMetrics(n_images=len(TEST_DATA), mean_AP=200 / 351)

    # test suite metrics - two
    test_suite_metrics_dup = eval.compute_test_suite_metrics(
        test_suite=TEST_SUITE,
        metrics=[
            (TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS),
            (TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS),
        ],
        configuration=config,
    )
    assert test_suite_metrics_dup == TestSuiteMetrics(n_images=len(TEST_DATA), mean_AP=200 / 351)
