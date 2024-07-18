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
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Optional
from typing import Tuple
from typing import Type

from kolena._api.v1.generic import TestRun as API
from kolena._utils.datatypes import DataObject
from kolena._utils.datatypes import get_args
from kolena._utils.datatypes import get_origin
from kolena._utils.pydantic_v1 import validate_arguments
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.validators import ValidatorConfig
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import TestCase
from kolena.workflow import TestSample
from kolena.workflow import TestSuite
from kolena.workflow._validators import get_data_object_field_types
from kolena.workflow._validators import validate_data_object_type
from kolena.workflow._validators import validate_scalar_data_object_type

# include for backwards compatibility -- definitions moved to ./plot.py
# noreorder
from kolena.workflow import AxisConfig  # noqa: F401
from kolena.workflow import Plot  # noqa: F401
from kolena.workflow import Curve  # noqa: F401
from kolena.workflow import CurvePlot  # noqa: F401
from kolena.workflow import Histogram  # noqa: F401
from kolena.workflow import BarPlot  # noqa: F401
from kolena.workflow import ConfusionMatrix  # noqa: F401


@dataclass(frozen=True, config=ValidatorConfig)
class MetricsTestSample(DataObject, metaclass=ABCMeta):
    """
    Test-sample-level metrics produced by an [`Evaluator`][kolena.workflow.Evaluator].

    This class should be subclassed with the relevant fields for a given workflow.

    Examples here may include the number of true positive detections on an image, the mean IOU of inferred polygon(s)
    with ground truth polygon(s), etc.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _validate_metrics_test_sample_type(cls)


@dataclass(frozen=True, config=ValidatorConfig)
class MetricsTestCase(DataObject, metaclass=ABCMeta):
    """
    Test-case-level metrics produced by an [`Evaluator`][kolena.workflow.Evaluator].

    This class should be subclassed with the relevant fields for a given workflow.

    Test-case-level metrics are aggregate metrics like [`precision`][kolena.workflow.metrics.precision],
    [`recall`][kolena.workflow.metrics.recall], and [`f1_score`][kolena.workflow.metrics.f1_score]. Any and all
    aggregate metrics that fit a workflow should be defined here.

    ## Nesting Aggregate Metrics

    `MetricsTestCase` supports nesting metrics objects, for e.g. reporting class-level metrics within a test case that
    contains multiple classes. Example usage:

    ```python
    @dataclass(frozen=True)
    class PerClassMetrics(MetricsTestCase):
        Class: str
        Precision: float
        Recall: float
        F1: float
        AP: float

    @dataclass(frozen=True)
    class TestCaseMetrics(MetricsTestCase):
        macro_Precision: float
        macro_Recall: float
        macro_F1: float
        mAP: float
        PerClass: List[PerClassMetrics]
    ```

    Any `str`-type fields (e.g. `Class` in the above example) will be used as identifiers when displaying nested metrics
    on Kolena. For best results, include at least one `str`-type field in nested metrics definitions.

    When comparing nested metrics from multiple models, an `int`-type column with any of the following names will be
    used for sample size in statistical significance calculations: `N`, `n`, `nTestSamples`, `n_test_samples`,
    `sampleSize`, `sample_size`, `SampleSize`.

    For a detailed overview of this feature, see the [:kolena-diagram-tree-16: Nesting Test Case
    Metrics](../../workflow/advanced-usage/nesting-test-case-metrics.md) advanced usage guide.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _validate_metrics_test_case_type(cls)


@dataclass(frozen=True, config=ValidatorConfig)
class MetricsTestSuite(DataObject, metaclass=ABCMeta):
    """
    Test-suite-level metrics produced by an [`Evaluator`][kolena.workflow.Evaluator].

    This class should be subclassed with the relevant fields for a given workflow.

    Test-suite-level metrics typically measure performance across test cases, e.g. penalizing variance across different
    subsets of a benchmark.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _validate_metrics_test_suite_type(cls)


@dataclass(frozen=True, config=ValidatorConfig)
class EvaluatorConfiguration(DataObject, metaclass=ABCMeta):
    """
    Configuration for an [`Evaluator`][kolena.workflow.Evaluator].

    Example evaluator configurations may specify:

    - Fixed confidence thresholds at which detections are discarded.
    - Different algorithms/strategies used to compute confidence thresholds
        (e.g. "accuracy optimal" for a classification-type workflow).
    """

    @abstractmethod
    def display_name(self) -> str:
        """
        The name to display for this configuration in Kolena. Must be implemented when extending
        [`EvaluatorConfiguration`][kolena.workflow.EvaluatorConfiguration].
        """
        raise NotImplementedError


class Evaluator(metaclass=ABCMeta):
    """
    An `Evaluator` transforms inferences into metrics.

    Metrics are computed at the individual test sample level ([`MetricsTestSample`][kolena.workflow.MetricsTestSample]),
    in aggregate at the test case level ([`MetricsTestCase`][kolena.workflow.MetricsTestCase]), and across populations
    at the test suite level ([`MetricsTestSuite`][kolena.workflow.MetricsTestSuite]).

    Test-case-level plots ([`Plot`][kolena.workflow.Plot]) may also be computed.

    :param configurations: The configurations at which to perform evaluation. Instance methods such as
        [`compute_test_sample_metrics`][kolena.workflow.Evaluator.compute_test_sample_metrics] are called once per test
        case per configuration.
    """

    configurations: List[EvaluatorConfiguration]
    """The configurations with which to perform evaluation, provided on instantiation."""

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, configurations: Optional[List[EvaluatorConfiguration]] = None):
        if configurations is not None and len(configurations) == 0:
            raise ValueError("empty configuration list provided, at least one configuration required or 'None'")
        self.configurations = configurations or []
        if len({configuration.display_name() for configuration in self.configurations}) < len(self.configurations):
            raise ValueError("all configurations must have distinct display names")

    def display_name(self) -> str:
        """The name to display for this evaluator in Kolena. Defaults to the name of this class."""
        return type(self).__name__

    @abstractmethod
    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> List[Tuple[TestSample, MetricsTestSample]]:
        """
        Compute metrics for every test sample in a test case, i.e. one
        [`MetricsTestSample`][kolena.workflow.MetricsTestSample] object for each of the provided test samples.

        Must be implemented.

        :param test_case: The [`TestCase`][kolena.workflow.TestCase] to which the provided test samples and ground
            truths belong.
        :param inferences: The test samples, ground truths, and inferences for all entries in a test case.
        :param configuration: The evaluator configuration to use. Empty for implementations that are not configured.
        :return: [`TestSample`][kolena.workflow.TestSample]-level metrics for each provided test sample.
        """
        raise NotImplementedError

    @abstractmethod
    def compute_test_case_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[MetricsTestSample],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> MetricsTestCase:
        """
        Compute aggregate metrics ([`MetricsTestCase`][kolena.workflow.MetricsTestCase]) across a test case.

        Must be implemented.

        :param test_case: The test case in question.
        :param inferences: The test samples, ground truths, and inferences for all entries in a test case.
        :param metrics: The [`TestSample`][kolena.workflow.TestSample]-level metrics computed by
            [`compute_test_sample_metrics`][kolena.workflow.Evaluator.compute_test_sample_metrics], provided
            in the same order as `inferences`.
        :param configuration: The evaluator configuration to use. Empty for implementations that are not configured.
        :return: [`TestCase`][kolena.workflow.TestCase]-level metrics for the provided test case.
        """
        raise NotImplementedError

    @validate_arguments(config=ValidatorConfig)
    def compute_test_case_plots(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        metrics: List[MetricsTestSample],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> Optional[List[Plot]]:
        """
        Optionally compute any number of plots to visualize the results for a test case.

        :param test_case: The test case in question
        :param inferences: The test samples, ground truths, and inferences for all entries in a test case.
        :param metrics: The [`TestSample`][kolena.workflow.TestSample]-level metrics computed by
            [`compute_test_sample_metrics`][kolena.workflow.Evaluator.compute_test_sample_metrics], provided
            in the same order as `inferences`.
        :param configuration: the evaluator configuration to use. Empty for implementations that are not configured.
        :return: Zero or more plots for this test case at this configuration.
        """
        return None  # not required

    @validate_arguments(config=ValidatorConfig)
    def compute_test_suite_metrics(
        self,
        test_suite: TestSuite,
        metrics: List[Tuple[TestCase, MetricsTestCase]],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> Optional[MetricsTestSuite]:
        """
        Optionally compute [`TestSuite`][kolena.workflow.TestSuite]-level metrics
        ([`MetricsTestSuite`][kolena.workflow.MetricsTestSuite]) across the provided `test_suite`.

        :param test_suite: The test suite in question.
        :param metrics: The [`TestCase`][kolena.workflow.TestCase]-level metrics computed by
            [`compute_test_case_metrics`][kolena.workflow.Evaluator.compute_test_case_metrics].
        :param configuration: The evaluator configuration to use. Empty for implementations that are not configured.
        :return: The [`TestSuite`][kolena.workflow.TestSuite]-level metrics for this test suite.
        """
        return None  # not required


def _validate_metrics_test_sample_type(metrics_test_sample_type: Type[MetricsTestSample]) -> None:
    # TODO: support special structure for ImagePair test sample types?
    validate_data_object_type(metrics_test_sample_type)


def _validate_metrics_test_case_type(metrics_test_case_type: Type[DataObject]) -> None:
    validate_scalar_data_object_type(metrics_test_case_type, supported_list_types=[MetricsTestCase])

    # validate that there is only one level of nesting
    for field_name, field_type in get_data_object_field_types(metrics_test_case_type).items():
        origin = get_origin(field_type)
        if origin is not list:  # only need to check lists, as MetricsTestCase is only allowed in lists
            continue
        # expand e.g. List[Union[MetricsA, MetricsB]] into [MetricsA, MetricsB]
        list_arg_types = [t for arg_type in get_args(field_type) for t in get_args(arg_type) or [arg_type]]
        for arg_type in list_arg_types:
            if arg_type is None:
                raise ValueError(f"Unsupported optional metrics object in field '{field_name}'")
            try:
                validate_scalar_data_object_type(arg_type)
            except ValueError:
                raise ValueError(f"Unsupported doubly-nested metrics object in field '{field_name}'")


def _validate_metrics_test_suite_type(metrics_test_suite_type: Type[DataObject]) -> None:
    validate_scalar_data_object_type(metrics_test_suite_type)


def _maybe_evaluator_configuration_to_api(
    configuration: Optional[EvaluatorConfiguration],
) -> Optional[API.EvaluatorConfiguration]:
    if configuration is None:
        return None
    return API.EvaluatorConfiguration(display_name=configuration.display_name(), configuration=configuration._to_dict())


def _maybe_display_name(configuration: Optional[EvaluatorConfiguration]) -> Optional[str]:
    if configuration is None:
        return None
    return configuration.display_name()


@validate_arguments(config=ValidatorConfig)
def _configuration_description(configuration: Optional[EvaluatorConfiguration]) -> str:
    display_name = _maybe_display_name(configuration)
    return f"(configuration: {display_name})" if display_name is not None else ""
