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

from .test_evaluator_single_class_test_sample_metrics import EXPECTED_COMPUTE_TEST_SAMPLE_METRICS
from .test_evaluator_single_class_test_sample_metrics import TEST_CONFIGURATIONS
from .test_evaluator_single_class_test_sample_metrics import TEST_DATA
from kolena._experimental.object_detection import ObjectDetectionEvaluator
from kolena._experimental.object_detection import TestCase
from kolena._experimental.object_detection import TestSuite
from kolena._experimental.object_detection.workflow import TestCaseMetricsSingleClass
from kolena._experimental.object_detection.workflow import TestSuiteMetrics
from tests.integration.helper import with_test_prefix


TEST_CASE_1 = TestCase(with_test_prefix("test_evaluator_single_class_suite_1"), reset=True)
TEST_CASE_2 = TestCase(with_test_prefix("test_evaluator_single_class_suite_2"), reset=True)


@pytest.mark.metrics
@pytest.mark.parametrize(
    "config_name, test_name, test_case_metrics, expected",
    [
        (
            "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
            "nothing",
            [
                (
                    TEST_CASE_1,
                    TestCaseMetricsSingleClass(
                        Objects=0,
                        Inferences=0,
                        TP=0,
                        FN=0,
                        FP=0,
                        Precision=0,
                        Recall=0,
                        F1=0,
                        AP=0,
                    ),
                ),
            ],
            TestSuiteMetrics(
                n_images=1,
                mean_AP=0,
            ),
        ),
        (
            "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0",
            "multiple bboxes in an image, varied iou",
            [
                (
                    TEST_CASE_1,
                    TestCaseMetricsSingleClass(
                        Objects=0,
                        Inferences=0,
                        TP=0,
                        FN=0,
                        FP=0,
                        Precision=0,
                        Recall=0,
                        F1=0,
                        AP=0.123,
                    ),
                ),
            ],
            TestSuiteMetrics(
                n_images=1,
                mean_AP=0.123,
            ),
        ),
    ],
)
def test__prebuilt__object__detection__single__class__compute__test__suite__metrics(
    config_name: str,
    test_name: str,
    test_case_metrics: List[Tuple[TestCase, TestCaseMetricsSingleClass]],
    expected: TestSuiteMetrics,
) -> None:
    config = TEST_CONFIGURATIONS[config_name]
    eval = ObjectDetectionEvaluator(configurations=[config])
    eval.compute_test_sample_metrics(
        test_case=test_case_metrics[0][0],
        inferences=TEST_DATA[test_name],
        configuration=config,
    )

    eval.compute_test_case_metrics(
        test_case=test_case_metrics[0][0],
        inferences=TEST_DATA[test_name],
        metrics=[metric for _, metric in EXPECTED_COMPUTE_TEST_SAMPLE_METRICS[config_name][test_name]],
        configuration=config,
    )

    result = eval.compute_test_suite_metrics(
        test_suite=TestSuite(
            with_test_prefix("test_evaluator_single_class_suite"),
            test_cases=[test_case_metrics[0][0]],
            reset=True,
        ),
        metrics=test_case_metrics,
        configuration=config,
    )

    assert expected == result


@pytest.mark.metrics
def test__prebuilt__object__detection__single__class__compute__multiple__test__suite__metric() -> None:
    config_name = "Threshold: Fixed(0.5), IoU: 0.5, confidence ≥ 0.0"
    test_1 = "multiple bboxes in an image, varied iou"
    test_2 = "nothing"
    config = TEST_CONFIGURATIONS[config_name]
    eval = ObjectDetectionEvaluator(configurations=[config])

    test_case_1 = TestCase(
        with_test_prefix(TEST_CASE_1),
        test_samples=[(ts, gt) for ts, gt, _ in TEST_DATA[test_1]],
        reset=True,
    )
    eval.compute_test_sample_metrics(
        test_case=test_case_1,
        inferences=TEST_DATA[test_1],
        configuration=config,
    )

    eval.compute_test_case_metrics(
        test_case=test_case_1,
        inferences=TEST_DATA[test_1],
        metrics=[EXPECTED_COMPUTE_TEST_SAMPLE_METRICS[config_name][test_1][0][1]],
        configuration=config,
    )

    test_case_2 = TestCase(
        with_test_prefix(TEST_CASE_2),
        test_samples=[(ts, gt) for ts, gt, _ in TEST_DATA[test_2]],
        reset=True,
    )
    eval.compute_test_sample_metrics(
        test_case=test_case_2,
        inferences=TEST_DATA[test_2],
        configuration=config,
    )

    eval.compute_test_case_metrics(
        test_case=test_case_2,
        inferences=TEST_DATA[test_2],
        metrics=[EXPECTED_COMPUTE_TEST_SAMPLE_METRICS[config_name][test_2][0][1]],
        configuration=config,
    )

    test_case_metrics = [
        (
            test_case_1,
            TestCaseMetricsSingleClass(
                Objects=0,
                Inferences=0,
                TP=0,
                FN=0,
                FP=0,
                Precision=0.0,
                Recall=0.0,
                F1=0.0,
                AP=0.3,
            ),
        ),
        (
            test_case_2,
            TestCaseMetricsSingleClass(
                Objects=0,
                Inferences=0,
                TP=0,
                FN=0,
                FP=0,
                Precision=0.0,
                Recall=0.0,
                F1=0.0,
                AP=0.7,
            ),
        ),
    ]

    result = eval.compute_test_suite_metrics(
        test_suite=TestSuite(
            with_test_prefix("test_evaluator_single_class_suite"),
            test_cases=[test_case_1, test_case_2],
            reset=True,
        ),
        metrics=test_case_metrics,
        configuration=config,
    )

    print(result)
    assert result == TestSuiteMetrics(
        n_images=2,
        mean_AP=0.5,
    )
