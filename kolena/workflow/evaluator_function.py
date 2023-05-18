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
import json
from abc import ABCMeta
from abc import abstractmethod
from dataclasses import field
from inspect import signature
from typing import Callable
from typing import Iterator
from typing import List
from typing import Optional
from typing import Tuple
from typing import TypeVar
from typing import Union

from pydantic.dataclasses import dataclass

from kolena._api.v1.generic import TestRun as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.state import is_client_initialized
from kolena._utils.validators import ValidatorConfig
from kolena.workflow import EvaluatorConfiguration
from kolena.workflow import GroundTruth as BaseGroundTruth
from kolena.workflow import Inference as BaseInference
from kolena.workflow import MetricsTestCase
from kolena.workflow import MetricsTestSample as BaseMetricsTestSample
from kolena.workflow import MetricsTestSuite
from kolena.workflow import Plot
from kolena.workflow import TestCase
from kolena.workflow import TestSample as BaseTestSample
from kolena.workflow.evaluator import _configuration_description

TestSample = TypeVar("TestSample", bound=BaseTestSample)
GroundTruth = TypeVar("GroundTruth", bound=BaseGroundTruth)
Inference = TypeVar("Inference", bound=BaseInference)
MetricsTestSample = TypeVar("MetricsTestSample", bound=BaseMetricsTestSample)


class TestCases(metaclass=ABCMeta):
    """
    Provides an iterator method for grouping test-sample-level metric results with the test cases that they belong to.
    """

    @abstractmethod
    def iter(
        self,
        test_samples: List[TestSample],
        ground_truths: List[GroundTruth],
        inferences: List[Inference],
        metrics_test_sample: List[MetricsTestSample],
    ) -> Iterator[Tuple[TestCase, List[TestSample], List[GroundTruth], List[Inference], List[MetricsTestSample]]]:
        """
        Matches test sample metrics to the corresponding test cases that they belong to.

        :param test_samples: all unique test samples within the test run, sequenced in the same order as the other
            parameters.
        :param ground_truths: ground truths corresponding to ``test_samples``, sequenced in the same order.
        :param inferences: inferences corresponding to ``test_samples``, sequenced in the same order.
        :param metrics_test_sample: test-sample-level metrics corresponding to ``test_samples``, sequenced in the
            same order.
        :return: an iterator that groups each test case in the test run to the lists of member test samples, inferences,
            and test-sample-level metrics.
        """
        raise NotImplementedError


@dataclass(frozen=True, config=ValidatorConfig)
class EvaluationResults:
    """
    A bundle of metrics computed for a test run grouped at the test-sample-level, test-case-level, and test-suite-level.
    Optionally includes :class:`kolena.workflow.Plot`s at the test-case-level.
    """

    metrics_test_sample: List[Tuple[BaseTestSample, BaseMetricsTestSample]]
    metrics_test_case: List[Tuple[TestCase, MetricsTestCase]]
    plots_test_case: List[Tuple[TestCase, List[Plot]]] = field(default_factory=list)
    metrics_test_suite: Optional[MetricsTestSuite] = None


ConfiguredEvaluatorFunction = Callable[
    [List[TestSample], List[GroundTruth], List[Inference], TestCases, EvaluatorConfiguration],
    Optional[EvaluationResults],
]
UnconfiguredEvaluatorFunction = Callable[
    [List[TestSample], List[GroundTruth], List[Inference], TestCases],
    Optional[EvaluationResults],
]
#: ``kolena.workflow.BasicEvaluatorFunction`` introduces a function based evaluator implementation that takes
#: the inferences for all test samples in a test suite and a :class:`kolena.workflow.TestCases` as input, and computes
#: the corresponding test-sample-level, test-case-level, and test-suite-level metrics (and optionally plots) as output.
#:
#: The control flow is in general more streamlined than with :class:`kolena.workflow.Evaluator`, but requires a couple
#: of assumptions to hold:
#:
#: - Test-sample-level metrics do not vary by test case
#: - Ground truths corresponding to a given test sample do not vary by test case
#:
#: This ``BasicEvaluatorFunction`` is provided to the test run at runtime, and is expected to have the
#: following signature:
#:
#: :param List[kolena.workflow.TestSample] test_samples: A list of distinct :class:`kolena.workflow.TestSample` values
#:     that correspond to all test samples in the test run.
#: :param List[kolena.workflow.GroundTruth] ground_truths: A list of :class:`kolena.workflow.GroundTruth` values
#:     corresponding to and sequenced in the same order as ``test_samples``.
#: :param List[kolena.workflow.Inference] inferences: A list of :class:`kolena.workflow.Inference` values corresponding
#:     to and sequenced in the same order as ``test_samples``.
#: :param TestCases test_cases: An instance of :class:`kolena.workflow.TestCases`, generally used to provide iteration
#:        groupings for evaluating test-case-level metrics.
#: :param EvaluatorConfiguration evaluator_configuration: The configuration to use when performing the evaluation.
#:     This parameter may be omitted in the function definition if running with no configuration.
#: :rtype: :class:`kolena.workflow.EvaluationResults`
#: :return: An object tracking the test-sample-level, test-case-level and test-suite-level metrics and plots for the
#:     input collection of test samples.
BasicEvaluatorFunction = Union[ConfiguredEvaluatorFunction, UnconfiguredEvaluatorFunction]


class _TestCases(TestCases):
    def __init__(
        self,
        test_case_membership: List[Tuple[TestCase, List[TestSample]]],
        test_run_id: int,
        n_configurations: int,
    ):
        self._test_case_membership = test_case_membership
        self._test_run_id = test_run_id
        self._wip_configuration: Optional[EvaluatorConfiguration] = None
        self._n_test_cases_and_configurations = max(n_configurations, 1) * len(self._test_case_membership)
        self._n_test_cases_processed = 0

    def iter(
        self,
        test_samples: List[TestSample],
        ground_truths: List[GroundTruth],
        inferences: List[Inference],
        metrics_test_sample: List[MetricsTestSample],
    ) -> Iterator[Tuple[TestCase, List[TestSample], List[GroundTruth], List[Inference], List[MetricsTestSample]]]:
        metrics_by_test_sample = {
            self._test_sample_key(ts): (ts, gt, inf, mts)
            for ts, gt, inf, mts in zip(test_samples, ground_truths, inferences, metrics_test_sample)
        }
        for tc, test_case_test_samples in self._test_case_membership:
            samples, gts, infs, metrics = [], [], [], []
            for ts in test_case_test_samples:
                test_sample_key = self._test_sample_key(ts)
                _, gt, inf, mts = metrics_by_test_sample[test_sample_key]
                samples.append(ts)
                gts.append(gt)
                infs.append(inf)
                metrics.append(mts)
            yield tc, samples, gts, infs, metrics
            self._n_test_cases_processed += 1
            self._update_progress(tc)

    def _update_progress(self, test_case: TestCase) -> None:
        if not is_client_initialized():
            return

        config_description = (
            f" {_configuration_description(self._wip_configuration)}" if self._wip_configuration else ""
        )
        message = f"Computed metrics for test case '{test_case.name}' (v{test_case.version}){config_description}"
        progress = self._n_test_cases_processed / self._n_test_cases_and_configurations

        log.info(message)
        request = API.UpdateMetricsStatusRequest(
            test_run_id=self._test_run_id,
            progress=progress,
            message=message,
        )
        res = krequests.put(
            endpoint_path=API.Path.UPDATE_METRICS_STATUS.value,
            data=json.dumps(dataclasses.asdict(request)),
        )
        krequests.raise_for_status(res)

    def _set_configuration(self, configuration: Optional[EvaluatorConfiguration]) -> None:
        self._wip_configuration = configuration
        config_description = (
            f" {_configuration_description(self._wip_configuration)}" if self._wip_configuration else ""
        )
        log.info(f"computing metrics{config_description}")

    @staticmethod
    def _test_sample_key(ts: TestSample) -> str:
        return json.dumps(ts._to_dict(), sort_keys=True)


def _is_configured(evaluator: BasicEvaluatorFunction) -> bool:
    param_values = list(signature(evaluator).parameters.values())
    return len(param_values) == 5 and issubclass(param_values[4].annotation, EvaluatorConfiguration)
