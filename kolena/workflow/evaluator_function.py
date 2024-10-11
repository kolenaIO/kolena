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
"""
Simplified interface for [`Evaluator`][kolena.workflow.Evaluator] implementations.
"""
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

from kolena._api.v1.generic import TestRun as API
from kolena._utils import krequests
from kolena._utils import log
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.state import is_client_uninitialized
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

        :param test_samples: All unique test samples within the test run, sequenced in the same order as the other
            parameters.
        :param ground_truths: Ground truths corresponding to `test_samples`, sequenced in the same order.
        :param inferences: Inferences corresponding to `test_samples`, sequenced in the same order.
        :param metrics_test_sample: Test-sample-level metrics corresponding to `test_samples`, sequenced in the
            same order.
        :return: Iterator that groups each test case in the test run to the lists of member test samples, inferences,
            and test-sample-level metrics.
        """
        raise NotImplementedError


@dataclass(frozen=True, config=ValidatorConfig)
class EvaluationResults:
    """
    A bundle of metrics computed for a test run grouped at the test-sample-level, test-case-level, and test-suite-level.
    Optionally includes [`Plot`s][kolena.workflow.Plot] at the test-case-level.
    """

    metrics_test_sample: List[Tuple[BaseTestSample, BaseMetricsTestSample]]
    """
    Sample-level metrics, extending [`MetricsTestSample`][kolena.workflow.MetricsTestSample], for every provided test
    sample.
    """

    metrics_test_case: List[Tuple[TestCase, MetricsTestCase]]
    """
    Aggregate metrics, extending [`MetricsTestCase`][kolena.workflow.MetricsTestCase], computed across each test case
    yielded from [`TestCases.iter`][kolena.workflow.TestCases.iter].
    """

    plots_test_case: List[Tuple[TestCase, List[Plot]]] = field(default_factory=list)
    """Optional test-case-level plots."""

    metrics_test_suite: Optional[MetricsTestSuite] = None
    """Optional test-suite-level metrics, extending [`MetricsTestSuite`][kolena.workflow.MetricsTestSuite]."""


ConfiguredEvaluatorFunction = Callable[
    [List[TestSample], List[GroundTruth], List[Inference], TestCases, EvaluatorConfiguration],
    Optional[EvaluationResults],
]
UnconfiguredEvaluatorFunction = Callable[
    [List[TestSample], List[GroundTruth], List[Inference], TestCases],
    Optional[EvaluationResults],
]
BasicEvaluatorFunction = Union[ConfiguredEvaluatorFunction, UnconfiguredEvaluatorFunction]
"""
`BasicEvaluatorFunction` provides a function-based evaluator interface that takes
the inferences for all test samples in a test suite and a [`TestCases`][kolena.workflow.TestCases] as input and computes
the corresponding test-sample-level, test-case-level, and test-suite-level metrics (and optionally plots) as output.

Example implementation, relying on `compute_per_sample` and `compute_aggregate` functions implemented elsewhere:

```python
def evaluate(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
    # configuration: EvaluatorConfiguration,  # uncomment when configuration is used
) -> EvaluationResults:
    # compute per-sample metrics for each test sample
    per_sample_metrics = [compute_per_sample(gt, inf) for gt, inf in zip(ground_truths, inferences)]

    # compute aggregate metrics across all test cases using `test_cases.iter(...)`
    aggregate_metrics: List[Tuple[TestCase, MetricsTestCase]] = []
    for test_case, *s in test_cases.iter(test_samples, ground_truths, inferences, per_sample_metrics):
        # subset of `test_samples`/`ground_truths`/`inferences`/`test_sample_metrics` in given test case
        tc_test_samples, tc_ground_truths, tc_inferences, tc_per_sample_metrics = s
        aggregate_metrics.append((test_case, compute_aggregate(tc_per_sample_metrics)))

    # if desired, compute and add `plots_test_case` and `metrics_test_suite`
    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, per_sample_metrics)),
        metrics_test_case=aggregate_metrics,
    )
```

The control flow is in general more streamlined than with [`Evaluator`][kolena.workflow.Evaluator], but requires a
couple of assumptions to hold:

- Test-sample-level metrics do not vary by test case
- Ground truths corresponding to a given test sample do not vary by test case

This `BasicEvaluatorFunction` is provided to the test run at runtime, and is expected to have the following signature:

:param List[TestSample] test_samples: A list of distinct [`TestSample`][kolena.workflow.TestSample] values
    that correspond to all test samples in the test run.
:param List[GroundTruth] ground_truths: A list of [`GroundTruth`][kolena.workflow.GroundTruth] values
    corresponding to and sequenced in the same order as `test_samples`.
:param List[Inference] inferences: A list of [`Inference`][kolena.workflow.Inference] values corresponding
    to and sequenced in the same order as `test_samples`.
:param TestCases test_cases: An instance of [`TestCases`][kolena.workflow.TestCases], used to provide iteration
    groupings for evaluating test-case-level metrics.
:param EvaluatorConfiguration evaluator_configuration: The
    [`EvaluatorConfiguration`][kolena.workflow.EvaluatorConfiguration] to use when performing the evaluation. This
    parameter may be omitted in the function definition for implementations that do not use any configuration object.
:rtype: EvaluationResults
:return: An [`EvaluationResults`][kolena.workflow.EvaluationResults] object tracking the test-sample-level,
    test-case-level and test-suite-level metrics and plots for the input collection of test samples.
"""


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
        if is_client_uninitialized():
            return

        config_description = (
            f" {_configuration_description(self._wip_configuration)}" if self._wip_configuration else ""
        )
        message = f"computed metrics for test case '{test_case.name}' (v{test_case.version}){config_description}"
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


def no_op_evaluator(
    test_samples: List[TestSample],
    ground_truths: List[GroundTruth],
    inferences: List[Inference],
    test_cases: TestCases,
) -> EvaluationResults:
    """
    A no-op implementation of the Kolena [`Evaluator`][kolena.workflow.Evaluator] that will bypass evaluation but
    make [`Inference`][kolena.workflow.Inference]s accessible in the platform.

    ```python
    from kolena.workflow import no_op_evaluator
    from kolena.workflow import test

    test(model, test_suite, no_op_evaluator)
    ```
    """
    test_sample_metrics = [BaseMetricsTestSample() for _ in test_samples]
    test_case_metrics = [
        (tc, MetricsTestCase())
        for tc, *_ in test_cases.iter(test_samples, ground_truths, inferences, test_sample_metrics)
    ]
    return EvaluationResults(
        metrics_test_sample=list(zip(test_samples, test_sample_metrics)),
        metrics_test_case=test_case_metrics,
    )
