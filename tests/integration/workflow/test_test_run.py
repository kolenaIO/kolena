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
import dataclasses
import math
import random
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from unittest.mock import patch

import numpy as np
import pytest
from pydantic.dataclasses import dataclass

from kolena._api.v1.generic import TestRun as TestRunAPI
from kolena._utils import krequests
from kolena.errors import RemoteError
from kolena.workflow import define_workflow
from kolena.workflow import EvaluationResults
from kolena.workflow import Evaluator
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import noop_evaluator
from kolena.workflow import test
from kolena.workflow import TestCase
from kolena.workflow import TestCases
from kolena.workflow import TestRun
from kolena.workflow import TestSample
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import ClassificationLabel
from tests.integration.helper import assert_sorted_list_equal
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix
from tests.integration.workflow.conftest import dummy_evaluator_function
from tests.integration.workflow.conftest import dummy_evaluator_function_with_config
from tests.integration.workflow.conftest import DummyConfiguration
from tests.integration.workflow.conftest import DummyEvaluator
from tests.integration.workflow.conftest import DummyMetricsTestCase
from tests.integration.workflow.conftest import DummyMetricsTestSample
from tests.integration.workflow.conftest import DummyTestSample
from tests.integration.workflow.conftest import Model
from tests.integration.workflow.conftest import TestCase as DummyTestCase
from tests.integration.workflow.conftest import TestSuite
from tests.integration.workflow.dummy import DUMMY_WORKFLOW
from tests.integration.workflow.dummy import DummyGroundTruth
from tests.integration.workflow.dummy import DummyInference
from tests.integration.workflow.dummy import GrabbagGroundTruth
from tests.integration.workflow.dummy import GrabbagInference
from tests.integration.workflow.dummy import GrabbagTestSample


def test__init(
    dummy_model: Model,
    dummy_test_suites: List[TestSuite],
) -> None:
    evaluator0 = DummyEvaluator(configurations=[DummyConfiguration(value="a")])

    test_run0 = TestRun(dummy_model, dummy_test_suites[0], evaluator0)
    test_run1 = TestRun(dummy_model, dummy_test_suites[0], evaluator0)
    assert test_run0._id == test_run1._id

    test_run2 = TestRun(dummy_model, dummy_test_suites[1], evaluator0)
    assert test_run0._id != test_run2._id

    # can tack additional configurations onto an existing test run
    evaluator1 = DummyEvaluator(configurations=[DummyConfiguration(value="a"), DummyConfiguration(value="b")])
    evaluator2 = DummyEvaluator(configurations=[DummyConfiguration(value="c")])

    test_run3 = TestRun(dummy_model, dummy_test_suites[0], evaluator1)
    test_run4 = TestRun(dummy_model, dummy_test_suites[0], evaluator2)
    assert test_run0._id == test_run3._id
    assert test_run0._id == test_run4._id

    # different evaluator creates different test run
    class DifferentDummyEvaluator(DummyEvaluator):
        ...  # display_name is updated to new class name

    evaluator1 = DifferentDummyEvaluator(configurations=[DummyConfiguration(value="a")])
    test_run5 = TestRun(dummy_model, dummy_test_suites[0], evaluator1)
    assert test_run0._id != test_run5._id


def test__load_test_samples(
    dummy_model: Model,
    dummy_test_suites: List[TestSuite],
    dummy_test_samples: List[DummyTestSample],
) -> None:
    evaluator = DummyEvaluator(configurations=[DummyConfiguration(value="test__load_test_samples")])
    test_run = TestRun(dummy_model, dummy_test_suites[0], evaluator)
    assert_sorted_list_equal(test_run.load_test_samples(), dummy_test_samples)


def test__load_test_samples__reset(
    dummy_model_reset: Model,
    dummy_test_suites: List[TestSuite],
    dummy_test_samples: List[DummyTestSample],
) -> None:
    model = dummy_model_reset
    test_suite = dummy_test_suites[0]
    evaluator = DummyEvaluator(configurations=[DummyConfiguration(value="test__load_test_samples__reset")])
    test(model, test_suite, evaluator)

    # when reset=False, loads no test samples after uploading inferences
    test_run = TestRun(model, test_suite, evaluator)
    assert test_run.load_test_samples() == []

    # when reset=True, loads all test samples after uploading inferences
    test_run = TestRun(model, test_suite, evaluator, reset=True)
    assert_sorted_list_equal(test_run.load_test_samples(), dummy_test_samples)


def test__upload_inferences(
    dummy_model: Model,
    dummy_test_suites: List[TestSuite],
) -> None:
    evaluator = DummyEvaluator(configurations=[DummyConfiguration(value="test__upload_inferences")])
    test_run = TestRun(dummy_model, dummy_test_suites[0], evaluator)

    all_test_samples = test_run.load_test_samples()
    test_run.upload_inferences([(ts, dummy_model.infer(ts)) for ts in all_test_samples[:3]])

    test_run = TestRun(dummy_model, dummy_test_suites[0], evaluator)
    assert_sorted_list_equal(test_run.load_test_samples(), all_test_samples[3:])

    with pytest.raises(RemoteError):  # these test samples have already been processed
        test_run.upload_inferences([(ts, dummy_model.infer(ts)) for ts in all_test_samples[:3]])


def test__upload_inferences__reset(
    dummy_model_reset: Model,
    dummy_test_suites: List[TestSuite],
) -> None:
    model = dummy_model_reset
    test_suite = dummy_test_suites[0]
    complete_test_case = test_suite.test_cases[0]
    evaluator = DummyEvaluator(configurations=[DummyConfiguration(value="test__upload_inferences__reset")])
    test(model, test_suite, evaluator)

    dummy_inference_score = 42

    def dummy_reset_infer() -> DummyInference:
        return DummyInference(dummy_inference_score)

    test_run = TestRun(model, test_suite, evaluator, reset=True)
    all_test_samples = test_run.load_test_samples()
    assert len(all_test_samples) > 0

    original_inferences = model.load_inferences(complete_test_case)
    test_run.upload_inferences([(ts, dummy_reset_infer()) for ts in all_test_samples])
    updated_inferences = model.load_inferences(complete_test_case)

    assert len(original_inferences) == len(updated_inferences)
    assert all([inf.score != dummy_inference_score for inf in [inf for (_, _, inf) in original_inferences]])
    assert all([inf.score == dummy_inference_score for inf in [inf for (_, _, inf) in updated_inferences]])


class TestTestDummyEvaluator(DummyEvaluator):
    ...  # display_name is updated to new class name


def test__test(
    dummy_model: Model,
    dummy_test_suites: List[TestSuite],
    dummy_test_samples: List[DummyTestSample],
) -> None:
    evaluator = TestTestDummyEvaluator(configurations=[DummyConfiguration(value="test__test")])

    test(dummy_model, dummy_test_suites[0], evaluator)

    test_run = TestRun(dummy_model, dummy_test_suites[0], evaluator)
    assert test_run.load_test_samples() == []

    with pytest.raises(RemoteError):  # already complete
        test_run.upload_inferences([(dummy_test_samples[0], dummy_model.infer(dummy_test_samples[0]))])


def test__test__reset(
    dummy_model_reset: Model,
    dummy_test_suites: List[TestSuite],
    dummy_test_samples: List[DummyTestSample],
) -> None:
    model = dummy_model_reset
    test_suite = dummy_test_suites[0]
    test_samples = dummy_test_samples
    evaluator = TestTestDummyEvaluator(configurations=[DummyConfiguration(value="test__test__reset")])

    test(model, test_suite, evaluator, reset=True)

    test_run = TestRun(model, test_suite, evaluator, reset=True)
    assert_sorted_list_equal(test_run.load_test_samples(), test_samples)


def test__test__mark_crashed(
    dummy_test_suites: List[TestSuite],
) -> None:
    def infer(_: DummyTestSample) -> DummyInference:
        raise RuntimeError

    class MarkCrashedDummyEvaluator(DummyEvaluator):
        ...

    name = with_test_prefix(f"{__file__}::test__test__mark_crashed model")
    model = Model(name=name, infer=infer)
    test_suite = dummy_test_suites[0]
    evaluator = MarkCrashedDummyEvaluator()
    test_run = TestRun(model, test_suite, evaluator)

    with patch("kolena.workflow.test_run.report_crash") as patched:
        with pytest.raises(RuntimeError):
            test_run.run()

    patched.assert_called_once_with(test_run._id, TestRunAPI.Path.MARK_CRASHED.value)


def test__evaluator__unconfigured(
    dummy_model: Model,
    dummy_test_suites: List[TestSuite],
) -> None:
    class UnconfiguredDummyEvaluator(DummyEvaluator):
        def compute_test_suite_metrics(
            self,
            test_suite: TestCase,
            metrics: List[Tuple[TestCase, MetricsTestCase]],
            configuration: Optional[DummyConfiguration] = None,
        ) -> None:
            return  # make sure omitting test suite metrics doesn't affect behavior

    evaluator = UnconfiguredDummyEvaluator()
    test_run = TestRun(dummy_model, dummy_test_suites[0], evaluator)

    test_run0 = TestRun(dummy_model, dummy_test_suites[0], evaluator)
    assert test_run._id == test_run0._id

    TestRun(dummy_model, dummy_test_suites[0], evaluator).run()


def test__handle_nan_inf() -> None:
    def compare_float(a: float, b: float) -> bool:
        if math.isnan(a) and math.isnan(b):
            return True
        return a == b

    @dataclass(frozen=True, order=True)
    class MyTestSample(TestSample):
        value: float

        def __eq__(self, other: Any) -> bool:
            return isinstance(other, type(self)) and compare_float(self.value, other.value)

    @dataclass(frozen=True, order=True)
    class MyGroundTruth(GroundTruth):
        value: float

        def __eq__(self, other: Any) -> bool:
            return isinstance(other, type(self)) and compare_float(self.value, other.value)

    @dataclass(frozen=True, order=True)
    class MyInference(Inference):
        value: float

        def __eq__(self, other: Any) -> bool:
            return isinstance(other, type(self)) and compare_float(self.value, other.value)

    @dataclass(frozen=True)
    class MyTestSampleMetrics(MetricsTestSample):
        value: float

        def __eq__(self, other: Any) -> bool:
            return isinstance(other, type(self)) and compare_float(self.value, other.value)

    @dataclass(frozen=True)
    class MyTestCaseMetrics(MetricsTestCase):
        min: float
        max: float
        avg: float

        def __eq__(self, other: Any) -> bool:
            return (
                isinstance(other, type(self))
                and compare_float(self.min, other.min)
                and compare_float(self.max, other.max)
                and compare_float(self.avg, other.avg)
            )

    class MyEvaluator(Evaluator):
        def compute_test_sample_metrics(
            self,
            test_case: TestCase,
            inferences: List[Tuple[MyTestSample, MyGroundTruth, MyInference]],
            configuration: Optional[EvaluatorConfiguration] = None,
        ) -> List[Tuple[MyTestSample, MyTestSampleMetrics]]:
            return [(sample, MyTestSampleMetrics(value=inference.value)) for sample, gt, inference in inferences]

        def compute_test_case_metrics(
            self,
            test_case: TestCase,
            inferences: List[Tuple[MyTestSample, MyGroundTruth, MyInference]],
            metrics: List[MetricsTestSample],
            configuration: Optional[EvaluatorConfiguration] = None,
        ) -> MyTestCaseMetrics:
            results = [inference.value for _, _, inference in inferences]
            return MyTestCaseMetrics(min=np.min(results), max=np.max(results), avg=np.average(results))

    name = with_test_prefix(f"{__file__}::test__handle_nan_inf")
    _, MyTestCase, MyTestSuite, MyModel = define_workflow("nan_workflow", MyTestSample, MyGroundTruth, MyInference)

    values = [-math.inf, math.inf, math.nan]

    # skip special values for test samples
    samples = [MyTestSample(value=i) for i in range(len(values))]
    # use special values in ground truths
    ground_truths = [MyGroundTruth(value=value) for value in values]
    test_samples = list(zip(samples, ground_truths))
    test_case = MyTestCase(f"{name} case one", test_samples=test_samples)
    test_suite = MyTestSuite(f"{name} suite one", test_cases=[test_case])
    # use special values in inferences
    dummy_inferences = [MyInference(value=value) for value in values]

    # match inference to test sample
    model = MyModel(f"{name} model", infer=lambda sample: dummy_inferences[int(sample.value)])
    evaluator = MyEvaluator()
    TestRun(model, test_suite, evaluator).run()

    inferences = model.load_inferences(test_case)
    assert_sorted_list_equal(inferences, list(zip(samples, ground_truths, dummy_inferences)))


def test__test__function_evaluator(
    dummy_model: Model,
    dummy_test_suites: List[TestSuite],
) -> None:
    test(dummy_model, dummy_test_suites[0], dummy_evaluator_function)
    TestRun(dummy_model, dummy_test_suites[0], dummy_evaluator_function)


def test__test__noop_evaluator(
    dummy_model: Model,
    dummy_test_suites: List[TestSuite],
) -> None:
    test(dummy_model, dummy_test_suites[0], noop_evaluator)
    TestRun(dummy_model, dummy_test_suites[0], noop_evaluator)


def test__test__function_evaluator__with_skip(
    dummy_model: Model,
    dummy_test_suites: List[TestSuite],
) -> None:
    config = [DummyConfiguration(value="skip")]
    test(dummy_model, dummy_test_suites[0], dummy_evaluator_function_with_config, config)
    TestRun(dummy_model, dummy_test_suites[0], dummy_evaluator_function_with_config, config)


@pytest.mark.depends(on=["test__test"])
def test__test__remote_evaluator(
    dummy_model: Model,
    dummy_test_suites: List[TestSuite],
) -> None:
    class MockResponse:
        def __init__(self, json_data: Any, status_code: int):
            self.json_data = json_data
            self.status_code = status_code

        def json(self) -> Any:
            return self.json_data

        def raise_for_status(self) -> None:
            if self.status_code >= 400:
                raise RuntimeError(f"bad status code: {self.status_code}")

    def mock_put(endpoint_path: str, data: Any = None, json: Any = None, **kwargs: Any) -> Any:
        if endpoint_path == TestRunAPI.Path.CREATE_OR_RETRIEVE.value:
            response = TestRunAPI.CreateOrRetrieveResponse(test_run_id=1000)
            return MockResponse(dataclasses.asdict(response), 200)
        return krequests.put(endpoint_path, data, json, **kwargs)

    with patch("kolena._utils.krequests.put", new=mock_put):
        with patch.object(TestRun, "_start_server_side_evaluation") as remote_patched:
            with patch.object(TestRun, "_perform_evaluation") as eval_patched:
                with patch.object(TestRun, "_perform_streamlined_evaluation") as streamlined_eval_patched:
                    TestRun(dummy_model, dummy_test_suites[0], evaluator=None).evaluate()
    remote_patched.assert_called_once()
    eval_patched.assert_not_called()
    streamlined_eval_patched.assert_not_called()


def test__dataobject_schema_mismatch() -> None:
    test_samples = [
        (
            GrabbagTestSample(
                locator=fake_locator(i, "dataobject"),
                value=i,
                bbox=BoundingBox(
                    top_left=(random.randint(0, 10), random.randint(0, 10)),
                    bottom_right=(random.randint(11, 20), random.randint(11, 20)),
                    value=i,
                    foo="bar",
                ),
            ),
            GrabbagGroundTruth(label=f"ground truth {i}", value=i, license="noncommercial"),
        )
        for i in range(10)
    ]
    # spotcheck extra field
    assert test_samples[1][0].bbox.foo == "bar"

    name = with_test_prefix(f"{__file__}::test__dataobject_schema_mismatch")
    _, GrabbagTestCase, GrabbagTestSuite, GrabbagModel = define_workflow(
        DUMMY_WORKFLOW.name,
        GrabbagTestSample,
        GrabbagGroundTruth,
        GrabbagInference,
    )
    test_case_name = f"{name} test case"
    test_case = GrabbagTestCase(test_case_name, test_samples=test_samples)

    loaded_test_samples = test_case.load_test_samples()

    assert sorted(loaded_test_samples) == sorted(test_samples)

    # dataclass without `extra = allow` should also work
    expected_alt_format = [
        (
            DummyTestSample(
                locator=test_sample.locator,
                value=test_sample.value,
                bbox=test_sample.bbox,
            ),
            DummyGroundTruth(label=ground_truth.label, value=ground_truth.value),
        )
        for test_sample, ground_truth in test_samples
    ]
    dummy_test_case = DummyTestCase(test_case_name)
    loaded_dummy_test_samples = dummy_test_case.load_test_samples()
    assert sorted(loaded_dummy_test_samples) == sorted(expected_alt_format)

    test_suite = GrabbagTestSuite(f"{name} test suite", test_cases=[test_case])
    dummy_inferences = [GrabbagInference(value=i, mask=ClassificationLabel(label="cat", source=i)) for i in range(10)]

    # match inference to test sample
    model = GrabbagModel(f"{name} model", infer=lambda sample: dummy_inferences[int(sample.value)])

    def evaluator(
        test_samples: List[GrabbagTestSample],
        ground_truths: List[GrabbagGroundTruth],
        inferences: List[GrabbagInference],
        test_cases: TestCases,
    ) -> Optional[EvaluationResults]:
        fixed_random_value = random.randint(0, 1_000_000_000)
        test_sample_to_metrics = [
            (ts, DummyMetricsTestSample(value=fixed_random_value, ignored=False)) for ts in test_samples
        ]
        metrics_test_sample = [metrics for _, metrics in test_sample_to_metrics]
        metrics_test_case: List[Tuple[TestCase, DummyMetricsTestCase]] = []
        for test_case, tc_test_samples, tc_gts, tc_infs, tc_ts_metrics in test_cases.iter(
            test_samples,
            ground_truths,
            inferences,
            metrics_test_sample,
        ):
            metrics = DummyMetricsTestCase(value=fixed_random_value, summary="foobar")
            metrics_test_case.append((test_case, metrics))

        return EvaluationResults(
            metrics_test_sample=test_sample_to_metrics,
            metrics_test_case=metrics_test_case,
        )

    TestRun(model, test_suite, evaluator).run()

    inferences = model.load_inferences(test_case)
    assert_sorted_list_equal(inferences, [(ts, gt, inf) for (ts, gt), inf in zip(test_samples, dummy_inferences)])
