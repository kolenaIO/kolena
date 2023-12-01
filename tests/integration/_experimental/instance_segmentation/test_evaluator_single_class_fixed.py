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

from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import ScoredLabeledPolygon
from kolena.workflow.plot import Curve
from kolena.workflow.plot import CurvePlot
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


instance_segmentation = pytest.importorskip(
    "kolena._experimental.instance_segmentation",
    reason="requires kolena[metrics] extra",
)
GroundTruth = instance_segmentation.GroundTruth
Inference = instance_segmentation.Inference
TestCase = instance_segmentation.TestCase
TestSample = instance_segmentation.TestSample
TestSuite = instance_segmentation.TestSuite
InstanceSegmentationEvaluator = instance_segmentation.InstanceSegmentationEvaluator
ThresholdConfiguration = instance_segmentation.ThresholdConfiguration
TestCaseMetricsSingleClass = instance_segmentation.TestCaseMetricsSingleClass
TestSampleMetricsSingleClass = instance_segmentation.TestSampleMetricsSingleClass
TestSuiteMetrics = instance_segmentation.TestSuiteMetrics


TEST_DATA: List[Tuple[TestSample, GroundTruth, Inference]] = [
    (
        TestSample(locator=fake_locator(112, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.9),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.8),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 0.7),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(113, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1.1, 1), (1.1, 2), (2, 1.1), (2, 2)], "a", 0.9),
                ScoredLabeledPolygon([(3.2, 3), (3.2, 4), (4, 3.2), (4, 4)], "a", 0.8),
                ScoredLabeledPolygon([(5.3, 5), (5.3, 6), (6, 5.3), (6, 6)], "a", 0.7),
                ScoredLabeledPolygon([(7.4, 7), (7.4, 8), (8, 7.4), (8, 8)], "a", 0.1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(114, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.6),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.5),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.4),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 0.1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(115, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 1),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 1),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 0.9),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 0.8),
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.9),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.9),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 0.8),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(116, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.4),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(117, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.9),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.9),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(118, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
            ],
            ignored_polygons=[
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.9),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.8),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.1),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 1),
            ],
        ),
    ),
    (
        TestSample(locator=fake_locator(119, "IS")),
        GroundTruth(
            polygons=[],
            ignored_polygons=[],
        ),
        Inference(
            polygons=[],
            ignored=True,
        ),
    ),
    (
        TestSample(locator=fake_locator(120, "IS")),
        GroundTruth(
            polygons=[
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
            ],
            ignored_polygons=[
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
        ),
        Inference(
            polygons=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.9),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.8),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.1),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 1),
            ],
            ignored=True,
        ),
    ),
]


EXPECTED_COMPUTE_TEST_SAMPLE_METRICS: List[Tuple[TestSample, TestSampleMetricsSingleClass]] = [
    (
        TestSample(locator=fake_locator(112, "IS"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.9),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.8),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 0.7),
            ],
            FP=[],
            FN=[],
            count_TP=4,
            count_FP=0,
            count_FN=0,
            has_TP=True,
            has_FP=False,
            has_FN=False,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.7,
            threshold=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(113, "IS"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledPolygon([(1.1, 1), (1.1, 2), (2, 1.1), (2, 2)], "a", 0.9),
            ],
            FP=[
                ScoredLabeledPolygon([(3.2, 3), (3.2, 4), (4, 3.2), (4, 4)], "a", 0.8),
                ScoredLabeledPolygon([(5.3, 5), (5.3, 6), (6, 5.3), (6, 6)], "a", 0.7),
            ],
            FN=[
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
            count_TP=1,
            count_FP=2,
            count_FN=3,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            ignored=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.7,
            threshold=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(114, "IS"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.6),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.5),
            ],
            FP=[],
            FN=[
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
            count_TP=2,
            count_FP=0,
            count_FN=2,
            has_TP=True,
            has_FP=False,
            has_FN=True,
            ignored=False,
            max_confidence_above_t=0.6,
            min_confidence_above_t=0.5,
            threshold=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(115, "IS"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.9),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.9),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 0.8),
            ],
            FP=[
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 1),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 1),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 0.9),
                ScoredLabeledPolygon([(0, 0), (0, 1), (1, 0), (1, 1)], "a", 0.8),
            ],
            FN=[],
            count_TP=4,
            count_FP=4,
            count_FN=0,
            has_TP=True,
            has_FP=True,
            has_FN=False,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=0.8,
            threshold=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(116, "IS"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
            ],
            FP=[],
            FN=[
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
            ],
            count_TP=1,
            count_FP=0,
            count_FN=3,
            has_TP=True,
            has_FP=False,
            has_FN=True,
            ignored=False,
            max_confidence_above_t=1,
            min_confidence_above_t=1,
            threshold=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(117, "IS"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.9),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.9),
            ],
            FP=[],
            FN=[
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
            ],
            count_TP=2,
            count_FP=0,
            count_FN=2,
            has_TP=True,
            has_FP=False,
            has_FN=True,
            ignored=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.9,
            threshold=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(118, "IS"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.9),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.8),
            ],
            FP=[],
            FN=[],
            count_TP=2,
            count_FP=0,
            count_FN=0,
            has_TP=True,
            has_FP=False,
            has_FN=False,
            ignored=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.8,
            threshold=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(119, "IS"), metadata={}),
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
            threshold=0.5,
        ),
    ),
    (
        TestSample(locator=fake_locator(120, "IS"), metadata={}),
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
            threshold=0.5,
        ),
    ),
]


EXPECTED_COMPUTE_TEST_CASE_METRICS = TestCaseMetricsSingleClass(
    Objects=26,
    Inferences=22,
    TP=16,
    FN=10,
    FP=6,
    nIgnored=2,
    Precision=16 / 22,
    Recall=16 / 26,
    F1=32 / 48,
    AP=200 / 343,
)


EXPECTED_F1_CURVE_PLOT = CurvePlot(
    title="F1-Score vs. Confidence Threshold",
    x_label="Confidence Threshold",
    y_label="F1-Score",
    curves=[
        Curve(
            label=None,
            x=[0, 0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            y=[40 / 53, 19 / 26, 18 / 25, 2 / 3, 30 / 47, 14 / 23, 13 / 22, 20 / 39, 6 / 31],
            extra={
                "Precision": [20 / 27, 19 / 26, 18 / 24, 16 / 22, 15 / 21, 0.7, 13 / 18, 10 / 13, 0.6],
                "Recall": [10 / 13, 19 / 26, 9 / 13, 8 / 13, 15 / 26, 7 / 13, 1 / 2, 10 / 26, 3 / 26],
            },
        ),
    ],
    x_config=None,
    y_config=None,
)


def assert_test_case_metrics_equals_expected(
    metrics: TestCaseMetricsSingleClass,
    other_metrics: TestCaseMetricsSingleClass,
) -> None:
    assert metrics.Objects == other_metrics.Objects
    assert metrics.Inferences == other_metrics.Inferences
    assert metrics.TP == other_metrics.TP
    assert metrics.FN == other_metrics.FN
    assert metrics.FP == other_metrics.FP
    assert pytest.approx(other_metrics.Precision, abs=1e-12) == metrics.Precision
    assert pytest.approx(other_metrics.Recall, abs=1e-12) == metrics.Recall
    assert pytest.approx(other_metrics.F1, abs=1e-12) == metrics.F1
    assert pytest.approx(other_metrics.AP, abs=1e-3) == metrics.AP


def assert_curves(
    curves: List[Curve],
    expected: List[Curve],
) -> None:
    assert len(curves) == len(expected)
    for curve, expectation in zip(curves, expected):
        print(curve, expectation)
        assert curve.label == expectation.label
        assert len(curve.x) == len(expectation.x)
        assert sum(abs(a - b) for a, b in zip(curve.x, expectation.x)) < 1e-12
        assert len(curve.y) == len(expectation.y)
        assert sum(abs(a - b) for a, b in zip(curve.y, expectation.y)) < 1e-12
        for extra_key in curve.extra.keys():
            assert sum(abs(a - b) for a, b in zip(curve.extra[extra_key], expectation.extra[extra_key])) < 1e-12


def assert_curve_plot_equal(
    plot: CurvePlot,
    expected: CurvePlot,
) -> None:
    assert plot.title == expected.title
    assert plot.x_label == expected.x_label
    assert plot.y_label == expected.y_label
    assert_curves(plot.curves, expected.curves)
    assert plot.x_config == expected.x_config
    assert plot.y_config == expected.y_config


@pytest.mark.metrics
def test__instance_segmentation__multiclass_evaluator__fixed() -> None:
    TEST_CASE_NAME = "single class IS test fixed"
    TEST_CASE = TestCase(with_test_prefix(TEST_CASE_NAME + " case"))
    TEST_SUITE = TestSuite(with_test_prefix(TEST_CASE_NAME + " suite"))
    config = ThresholdConfiguration(
        threshold_strategy=0.5,
    )

    eval = InstanceSegmentationEvaluator(configurations=[config])

    test_sample_metrics = eval.compute_test_sample_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA,
        configuration=config,
    )

    assert config.display_name() not in eval.evaluator.threshold_cache
    assert len(eval.evaluator.matchings_by_test_case) != 0
    assert len(eval.evaluator.matchings_by_test_case[config.display_name()]) != 0
    num_of_ignored = sum([1 for _, _, inf in TEST_DATA if inf.ignored])
    assert (
        len(eval.evaluator.matchings_by_test_case[config.display_name()][TEST_CASE.name])
        == len(TEST_DATA) - num_of_ignored
    )
    assert test_sample_metrics == EXPECTED_COMPUTE_TEST_SAMPLE_METRICS

    test_case_metrics = eval.compute_test_case_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA,
        metrics=[pair[1] for pair in EXPECTED_COMPUTE_TEST_SAMPLE_METRICS],
        configuration=config,
    )
    assert TEST_CASE.name in eval.evaluator.locators_by_test_case
    assert len(eval.evaluator.locators_by_test_case[TEST_CASE.name]) == len(TEST_DATA)
    assert_test_case_metrics_equals_expected(test_case_metrics, EXPECTED_COMPUTE_TEST_CASE_METRICS)

    # test case plots only use the cached values
    plots = eval.compute_test_case_plots(
        test_case=TEST_CASE,
        inferences=[],
        metrics=[],
        configuration=config,
    )
    assert_curve_plot_equal(plots[1], EXPECTED_F1_CURVE_PLOT)

    # test suite metrics - one
    test_suite_metrics = eval.compute_test_suite_metrics(
        test_suite=TEST_SUITE,
        metrics=[
            (TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS),
        ],
        configuration=config,
    )
    assert test_suite_metrics == TestSuiteMetrics(
        n_images=len(TEST_DATA),
        mean_AP=200 / 343,
        threshold=config.threshold_strategy,
    )

    # test suite metrics - two
    test_suite_metrics_dup = eval.compute_test_suite_metrics(
        test_suite=TEST_SUITE,
        metrics=[
            (TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS),
            (TEST_CASE, EXPECTED_COMPUTE_TEST_CASE_METRICS),
        ],
        configuration=config,
    )
    assert test_suite_metrics_dup == TestSuiteMetrics(
        n_images=len(TEST_DATA),
        mean_AP=200 / 343,
        threshold=config.threshold_strategy,
    )
