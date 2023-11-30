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
from tests.integration._experimental.instance_segmentation.test_evaluator_single_class_fixed import (
    assert_curve_plot_equal,
)
from tests.integration._experimental.instance_segmentation.test_evaluator_single_class_fixed import (
    assert_test_case_metrics_equals_expected,
)
from tests.integration._experimental.instance_segmentation.test_evaluator_single_class_fixed import TEST_DATA
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix


instance_segmentation = pytest.importorskip(
    "kolena._experimental.instance_segmentation",
    reason="requires kolena[metrics] extra",
)
InstanceSegmentationEvaluator = instance_segmentation.InstanceSegmentationEvaluator
TestSample = instance_segmentation.TestSample
TestCase = instance_segmentation.TestCase
ThresholdConfiguration = instance_segmentation.ThresholdConfiguration
TestCaseMetricsSingleClass = instance_segmentation.TestCaseMetricsSingleClass
TestSampleMetricsSingleClass = instance_segmentation.TestSampleMetricsSingleClass


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
            max_confidence_above_t=1.0,
            min_confidence_above_t=0.7,
            threshold=0.1,
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
                ScoredLabeledPolygon([(7.4, 7), (7.4, 8), (8, 7.4), (8, 8)], "a", 0.1),
            ],
            FN=[
                LabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a"),
                LabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
            ],
            count_TP=1,
            count_FP=3,
            count_FN=3,
            has_TP=True,
            has_FP=True,
            has_FN=True,
            ignored=False,
            max_confidence_above_t=0.9,
            min_confidence_above_t=0.1,
            threshold=0.1,
        ),
    ),
    (
        TestSample(locator=fake_locator(114, "IS"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 0.6),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.5),
                ScoredLabeledPolygon([(5, 5), (5, 6), (6, 5), (6, 6)], "a", 0.4),
                ScoredLabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a", 0.1),
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
            max_confidence_above_t=0.6,
            min_confidence_above_t=0.1,
            threshold=0.1,
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
            threshold=0.1,
        ),
    ),
    (
        TestSample(locator=fake_locator(116, "IS"), metadata={}),
        TestSampleMetricsSingleClass(
            TP=[
                ScoredLabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a", 1),
                ScoredLabeledPolygon([(3, 3), (3, 4), (4, 3), (4, 4)], "a", 0.4),
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
            max_confidence_above_t=1,
            min_confidence_above_t=0.4,
            threshold=0.1,
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
                LabeledPolygon([(1, 1), (1, 2), (2, 1), (2, 2)], "a"),
                LabeledPolygon([(7, 7), (7, 8), (8, 7), (8, 8)], "a"),
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
            threshold=0.1,
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
            threshold=0.1,
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
            threshold=0.1,
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
            threshold=0.1,
        ),
    ),
]


EXPECTED_COMPUTE_TEST_CASE_METRICS = TestCaseMetricsSingleClass(
    Objects=26,
    Inferences=26,
    TP=19,
    FN=7,
    FP=7,
    nIgnored=2,
    Precision=19 / 26,
    Recall=19 / 26,
    F1=19 / 26,
    AP=375 / 676,
)


EXPECTED_F1_CURVE_PLOT = CurvePlot(
    title="F1-Score vs. Confidence Threshold",
    x_label="Confidence Threshold",
    y_label="F1-Score",
    curves=[
        Curve(
            label=None,
            x=[0.1, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
            y=[19 / 26, 18 / 25, 2 / 3, 30 / 47, 14 / 23, 13 / 22, 20 / 39, 6 / 31],
            extra={
                "Precision": [19 / 26, 18 / 24, 16 / 22, 15 / 21, 0.7, 13 / 18, 10 / 13, 0.6],
                "Recall": [19 / 26, 9 / 13, 8 / 13, 15 / 26, 7 / 13, 1 / 2, 10 / 26, 3 / 26],
            },
        ),
    ],
    x_config=None,
    y_config=None,
)


@pytest.mark.metrics
def test__instance_segmentation__multiclass_evaluator__f1_optimal() -> None:
    TEST_CASE_NAME = "single class IS test fixed"
    TEST_CASE = TestCase(with_test_prefix(TEST_CASE_NAME + " case"))

    config = ThresholdConfiguration(
        iou_threshold=0.5,
        min_confidence_score=0.1,
    )

    eval = InstanceSegmentationEvaluator(configurations=[config])

    test_sample_metrics = eval.compute_test_sample_metrics(
        test_case=TEST_CASE,
        inferences=TEST_DATA,
        configuration=config,
    )

    assert config.display_name() in eval.evaluator.threshold_cache
    assert eval.evaluator.threshold_cache[config.display_name()] == 0.1
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

    # test suite behaviour is consistent with fixed evaluator
