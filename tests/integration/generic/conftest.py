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
import random
import uuid
from typing import List
from typing import Optional
from typing import Tuple

import pytest
from pydantic.dataclasses import dataclass

from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import Evaluator
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.evaluator import Plot
from kolena.workflow.evaluator_function import EvaluationResults
from kolena.workflow.evaluator_function import TestCases
from tests.integration.generic.dummy import DummyGroundTruth
from tests.integration.generic.dummy import DummyInference
from tests.integration.generic.dummy import DummyTestSample
from tests.integration.generic.dummy import Model
from tests.integration.generic.dummy import TestCase
from tests.integration.generic.dummy import TestSuite
from tests.integration.helper import fake_locator
from tests.integration.helper import with_test_prefix

N_DUMMY = 10


def dummy_test_sample(index: int, directory: str = "default") -> DummyTestSample:
    # TODO: this type ignore is ridiculous and will be annoying for customers extending our classes
    #  mypy: error: Unexpected keyword argument "locator" for "DummyTestSample"
    return DummyTestSample(  # type: ignore
        locator=fake_locator(index, directory),
        value=index,
        bbox=BoundingBox(
            top_left=(random.randint(0, 10), random.randint(0, 10)),
            bottom_right=(random.randint(11, 20), random.randint(11, 20)),
        ),
        metadata={},
    )


def dummy_ground_truth(index: int) -> DummyGroundTruth:
    return DummyGroundTruth(label=f"ground truth {index}", value=index)


@pytest.fixture(scope="package", autouse=True)
def init_kolena(with_init: None) -> None:
    ...


@pytest.fixture(scope="package")
def dummy_test_samples() -> List[DummyTestSample]:
    directory = str(uuid.uuid4())
    return [dummy_test_sample(i, directory) for i in range(N_DUMMY)]


@pytest.fixture(scope="package")
def dummy_ground_truths() -> List[DummyGroundTruth]:
    return [dummy_ground_truth(i) for i in range(N_DUMMY)]


def dummy_inference() -> DummyInference:
    return DummyInference(score=random.random())


@pytest.fixture(scope="package")
def dummy_model() -> Model:
    name = with_test_prefix(f"{__file__}::modelA")
    return Model(name=name, infer=lambda _: dummy_inference())


@pytest.fixture(scope="package")
def dummy_model_reset() -> Model:
    name = with_test_prefix(f"{__file__}::modelB for reset")
    return Model(name=name, infer=lambda _: dummy_inference())


@pytest.fixture(scope="package")
def dummy_test_suites(
    dummy_test_samples: List[DummyTestSample],
    dummy_ground_truths: List[DummyGroundTruth],
) -> List[TestSuite]:
    all_test_samples = list(zip(dummy_test_samples, dummy_ground_truths))
    return [
        TestSuite(
            with_test_prefix(f"{__file__}::TestSuite::A"),
            test_cases=[
                TestCase(with_test_prefix(f"{__file__}::TestCase::A"), test_samples=all_test_samples),
                TestCase(with_test_prefix(f"{__file__}::TestCase::B"), test_samples=all_test_samples[:5]),
                TestCase(with_test_prefix(f"{__file__}::TestCase::C"), test_samples=all_test_samples[5:]),
            ],
        ),
        TestSuite(
            with_test_prefix(f"{__file__}::TestSuite::B"),
        ),
    ]


@dataclass(frozen=True)
class DummyConfiguration(EvaluatorConfiguration):
    value: str

    def display_name(self) -> str:
        return f"{type(self).__name__}(value={self.value})"


@dataclass(frozen=True)
class DummyMetricsTestSample(MetricsTestSample):
    value: int


@dataclass(frozen=True)
class DummyMetricsTestCase(MetricsTestCase):
    value: int


@dataclass(frozen=True)
class DummyMetricsTestSuite(MetricsTestSuite):
    value: int


class DummyEvaluator(Evaluator):
    def __init__(self, configurations: Optional[List[DummyConfiguration]] = None):
        super().__init__(configurations)
        self.fixed_random_value = random.randint(0, 1_000_000_000)

    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[DummyTestSample, DummyGroundTruth, DummyInference]],
        configuration: Optional[DummyConfiguration] = None,
    ) -> List[Tuple[DummyTestSample, DummyMetricsTestSample]]:
        return [(ts, DummyMetricsTestSample(value=self.fixed_random_value)) for ts, *_ in inferences]

    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[DummyTestSample, DummyGroundTruth, DummyInference]],
        metrics: List[MetricsTestSample],
        configuration: Optional[DummyConfiguration] = None,
    ) -> DummyMetricsTestCase:
        return DummyMetricsTestCase(value=self.fixed_random_value)

    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[DummyTestSample, DummyGroundTruth, DummyInference]],
        metrics: List[MetricsTestSample],
        configuration: Optional[DummyConfiguration] = None,
    ) -> Optional[List[Plot]]:
        return [
            CurvePlot(title="first", x_label="x", y_label="y", curves=[Curve(label="l", x=[1, 2, 3], y=[4, 5, 6])]),
            CurvePlot(title="second", x_label="x", y_label="y", curves=[Curve(label="l", x=[3, 2, 1], y=[6, 5, 4])]),
        ]

    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, MetricsTestCase]],
        configuration: Optional[DummyConfiguration] = None,
    ) -> Optional[DummyMetricsTestSuite]:
        return DummyMetricsTestSuite(value=self.fixed_random_value)


DummyTestSampleInference = Tuple[DummyTestSample, DummyGroundTruth, DummyInference]


def dummy_evaluator_function(
    test_samples: List[DummyTestSample],
    ground_truths: List[DummyGroundTruth],
    inferences: List[DummyInference],
    test_cases: TestCases,
) -> Optional[EvaluationResults]:
    fixed_random_value = random.randint(0, 1_000_000_000)
    test_sample_to_metrics = [(ts, DummyMetricsTestSample(value=fixed_random_value)) for ts in test_samples]
    metrics_test_sample = [metrics for _, metrics in test_sample_to_metrics]
    metrics_test_case: List[Tuple[TestCase, DummyMetricsTestCase]] = []
    plots_test_case: List[Tuple[TestCase, List[Plot]]] = []
    for test_case, tc_test_samples, tc_gts, tc_infs, tc_ts_metrics in test_cases.iter(
        test_samples,
        ground_truths,
        inferences,
        metrics_test_sample,
    ):
        metrics = DummyMetricsTestCase(value=fixed_random_value)
        plots = [
            CurvePlot(title="first", x_label="x", y_label="y", curves=[Curve(label="l", x=[1, 2, 3], y=[4, 5, 6])]),
            CurvePlot(title="second", x_label="x", y_label="y", curves=[Curve(label="l", x=[3, 2, 1], y=[6, 5, 4])]),
        ]
        metrics_test_case.append((test_case, metrics))
        plots_test_case.append((test_case, plots))
    metrics_test_suite = DummyMetricsTestSuite(value=fixed_random_value)

    return EvaluationResults(
        metrics_test_sample=test_sample_to_metrics,
        metrics_test_case=metrics_test_case,
        plots_test_case=plots_test_case,
        metrics_test_suite=metrics_test_suite,
    )


def dummy_evaluator_function_with_config(
    test_samples: List[DummyTestSample],
    ground_truths: List[DummyGroundTruth],
    inferences: List[DummyInference],
    test_cases: TestCases,
    config: DummyConfiguration,
) -> Optional[EvaluationResults]:
    if config.value == "skip":
        return None
    return dummy_evaluator_function(test_samples, ground_truths, inferences, test_cases)
