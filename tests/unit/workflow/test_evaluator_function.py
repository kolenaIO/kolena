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

from kolena._api.v1.core import TestCase as CoreAPI
from kolena.workflow import define_workflow
from kolena.workflow import GroundTruth
from kolena.workflow import Image
from kolena.workflow import Inference
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow.evaluator_function import _TestCases
from kolena.workflow.evaluator_function import EvaluationResults


class DummyTestSample(Image):
    ...


class DummyGroundTruth(GroundTruth):
    is_live: bool


class DummyInference(Inference):
    confidence: float


class DummyTestSampleMetrics(MetricsTestSample):
    is_correct: bool


class DummyTestCaseMetrics(MetricsTestCase):
    percent_correct: float


TestSampleInference = Tuple[DummyTestSample, DummyGroundTruth, DummyInference]


DUMMY_WORKFLOW, TestCase, TestSuite, Model = define_workflow(
    name="dummy-workflow",
    test_sample_type=DummyTestSample,
    ground_truth_type=DummyGroundTruth,
    inference_type=DummyInference,
)


def test__test_cases() -> None:
    test_case_1 = TestCase._create_from_data(
        CoreAPI.EntityData(
            id=1,
            name="test_case_1",
            version=1,
            description="",
            workflow=DUMMY_WORKFLOW.name,
        ),
    )
    test_case_2 = TestCase._create_from_data(
        CoreAPI.EntityData(
            id=2,
            name="test_case_2",
            version=1,
            description="",
            workflow=DUMMY_WORKFLOW.name,
        ),
    )

    raw_results = [
        (
            DummyTestSample(locator=f"s3://dummy/{i}.jpg"),
            DummyGroundTruth(is_live=i % 2 == 0),
            DummyInference(confidence=i / 10),
            DummyTestSampleMetrics(is_correct=i % 3 == 0),
        )
        for i in range(3)
    ]

    def as_test_sample_results(
        results: List[Tuple[DummyTestSample, DummyGroundTruth, DummyInference, DummyTestSampleMetrics]],
    ) -> Tuple[List[DummyTestSample], List[DummyGroundTruth], List[DummyInference], List[DummyTestSampleMetrics]]:
        samples, gts, infs, metrics = [], [], [], []
        for sample, gt, inf, tsm in results:
            samples.append(sample)
            gts.append(gt)
            infs.append(inf)
            metrics.append(tsm)
        return samples, gts, infs, metrics

    test_case_membership = [
        (test_case_1, [ts for ts, *_ in raw_results[:2]]),
        (test_case_2, [ts for ts, *_ in raw_results[1:]]),
    ]
    test_run_id = 1
    test_samples, ground_truths, inferences, test_sample_metrics = as_test_sample_results(raw_results)
    test_cases = _TestCases(
        test_case_membership,
        test_run_id,
        0,
    )

    test_case_iterator = test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics)
    assert next(test_case_iterator) == (test_case_1, *as_test_sample_results(raw_results[:2]))
    assert next(test_case_iterator) == (test_case_2, *as_test_sample_results(raw_results[1:]))
    with pytest.raises(StopIteration):
        next(test_case_iterator)


def test__evaluation_results() -> None:
    test_sample_results = [
        (
            DummyTestSample(locator=f"s3://dummy/{i}.jpg"),
            DummyGroundTruth(is_live=i % 2 == 0),
            DummyInference(confidence=i / 10),
            DummyTestSampleMetrics(is_correct=i % 3 == 0),
        )
        for i in range(3)
    ]
    test_case_1 = TestCase._create_from_data(
        CoreAPI.EntityData(
            id=1,
            name="test_case_1",
            version=1,
            description="",
            workflow=DUMMY_WORKFLOW.name,
        ),
    )

    test_sample_metrics = [(ts, tsm) for ts, _, _, tsm in test_sample_results]
    test_case_metrics = [(test_case_1, DummyTestCaseMetrics(percent_correct=0.5))]
    got = EvaluationResults(
        metrics_test_sample=test_sample_metrics,
        metrics_test_case=test_case_metrics,
    )
    assert got.metrics_test_sample == test_sample_metrics
    assert got.metrics_test_case == test_case_metrics
    assert got.plots_test_case == []
    assert got.metrics_test_suite is None
