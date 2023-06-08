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
from abc import ABCMeta
from abc import abstractmethod
from typing import Any
from typing import List
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Type
from typing import Union

import numpy as np
from pydantic import validate_arguments
from pydantic.dataclasses import dataclass
from pydantic.typing import Literal

from kolena._api.v1.generic import TestRun as API
from kolena._utils.datatypes import get_args
from kolena._utils.datatypes import get_origin
from kolena._utils.validators import ValidatorConfig
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import TestCase
from kolena.workflow import TestSample
from kolena.workflow import TestSuite
from kolena.workflow._datatypes import DataObject
from kolena.workflow._datatypes import DataType
from kolena.workflow._datatypes import TypedDataObject
from kolena.workflow._validators import get_data_object_field_types
from kolena.workflow._validators import validate_data_object_type
from kolena.workflow._validators import validate_scalar_data_object_type


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

    Test-case-level metrics are aggregate metrics like Precision, Recall, and F1 score. Any and all aggregate metrics
    that fit a workflow should be defined here.

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


NumberSeries = Sequence[Union[float, int]]
"""A sequence of numeric values."""

NullableNumberSeries = Sequence[Union[float, int, None]]
"""A sequence of numeric values or `None`."""


class _PlotType(DataType):
    CURVE = "CURVE"
    CONFUSION_MATRIX = "CONFUSION_MATRIX"
    HISTOGRAM = "HISTOGRAM"
    BAR = "BAR"

    @staticmethod
    def _data_category() -> str:
        return "PLOT"


@dataclass(frozen=True, config=ValidatorConfig)
class AxisConfig(DataObject):
    """Configuration for the format of a given axis on a plot."""

    type: Literal["linear", "log"]
    """Type of axis to display. Supported options are `linear` and `log`."""


@dataclass(frozen=True, config=ValidatorConfig)
class Plot(TypedDataObject[_PlotType], metaclass=ABCMeta):
    """A data visualization shown when exploring model results in the web platform."""


@dataclass(frozen=True, config=ValidatorConfig)
class Curve(DataObject):
    """A single series on a [`CurvePlot`][kolena.workflow.CurvePlot]."""

    x: NumberSeries
    """The `x` coordinates of this curve. Length must match the provided `y` coordinates."""

    y: NumberSeries
    """The `y` coordinates of this curve. Length must match the provided `x` coordinates."""

    label: Optional[str] = None
    """
    Optionally specify an additional label (in addition to the associated test case) to apply to this curve, for use
    when e.g. there are multiple curves generated per test case.
    """

    def __post_init_post_parse__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(
                f"Series 'x' (length: {len(self.x)}) and 'y' (length: {len(self.y)}) have different lengths",
            )


@dataclass(frozen=True, config=ValidatorConfig)
class CurvePlot(Plot):
    """
    A plot visualizing one or more curves per test case.

    Examples include Receiver Operating Characteristic (ROC) curves, Precision versus Recall (PR) curves,
    Detection-Error Tradeoff (DET) curves, etc.
    """

    title: str
    """The title for the plot."""

    x_label: str
    """The label describing the plot's `x` axis."""

    y_label: str
    """The label describing the plot's `y` axis."""

    curves: List[Curve]
    """
    A test case may generate zero or more curves on a given plot. However, under most circumstances, a single curve
    per test case is desirable.
    """

    x_config: Optional[AxisConfig] = None
    """
    Custom options to allow for control over the display of the plot `x` axis. See
    [`AxisConfig`][kolena.workflow.AxisConfig] for details.
    """

    y_config: Optional[AxisConfig] = None
    """
    Custom options to allow for control over the display of the plot `y` axis. See
    [`AxisConfig`][kolena.workflow.AxisConfig] for details.
    """

    @staticmethod
    def _data_type() -> _PlotType:
        return _PlotType.CURVE


@dataclass(frozen=True, config=ValidatorConfig)
class Histogram(Plot):
    """
    A plot visualizing distribution of one or more continuous values, e.g. distribution of an error metric across all
    samples within a test case.

    For visualization of discrete values, see [`BarPlot`][kolena.workflow.BarPlot].
    """

    title: str
    """The title for the plot."""

    x_label: str
    """The label describing the plot's `x` axis."""

    y_label: str
    """The label describing the plot's `y` axis."""

    buckets: NumberSeries
    """
    A Histogram requires intervals to bucket the data. For `n` buckets, `n+1` consecutive bounds must be specified in
    increasing order.
    """

    frequency: Union[NumberSeries, Sequence[NumberSeries]]
    """
    For `n` buckets, there are `n` frequencies corresponding to the height of each bucket. The frequency at index `i`
    corresponds to the bucket with bounds (`i`, `i+1`) in `buckets`.

    To specify multiple distributions for a given test case, multiple frequency series can be provided, corresponding
    e.g. to the distribution for a given class within a test case, with name specified in `labels`.

    Specify a list of labels corresponding to the different `frequency` series when multiple series are provided.
    Can be omitted when a single `frequency` series is provided.
    """

    labels: Optional[List[str]] = None
    """Specify the label corresponding to a given distribution when multiple are specified in `frequency`."""

    x_config: Optional[AxisConfig] = None
    """
    Custom options to allow for control over the display of the plot `x` axis. See
    [`AxisConfig`][kolena.workflow.AxisConfig] for details.
    """

    y_config: Optional[AxisConfig] = None
    """
    Custom options to allow for control over the display of the plot `y` axis. See
    [`AxisConfig`][kolena.workflow.AxisConfig] for details.
    """

    def __post_init_post_parse__(self) -> None:
        n_buckets = len(self.buckets)
        if n_buckets < 2:
            raise ValueError(f"Minimum 2 entries required in 'buckets' series (length: {n_buckets})")
        buckets_arr = np.array(self.buckets)
        if not np.all(buckets_arr[1:] > buckets_arr[:-1]):  # validate strictly increasing
            raise ValueError("Invalid 'buckets' series: series must be strictly increasing")

        frequency_arr = np.array(self.frequency)
        actual_frequency_shape = np.shape(frequency_arr)
        if len(actual_frequency_shape) > 1 and self.labels is None:
            raise ValueError("'labels' are required when multiple 'frequency' series are provided")

        n_labels = len(self.labels or [])
        expected_frequency_shape = (n_labels, n_buckets - 1) if n_labels > 0 else (n_buckets - 1,)
        if actual_frequency_shape != expected_frequency_shape:
            raise ValueError(f"Invalid 'frequency' shape {actual_frequency_shape}: expected {expected_frequency_shape}")

    @staticmethod
    def _data_type() -> _PlotType:
        return _PlotType.HISTOGRAM


@dataclass(frozen=True, config=ValidatorConfig)
class BarPlot(Plot):
    """A plot visualizing a set of bars per test case."""

    title: str
    """The plot title."""

    x_label: str
    """Axis label for the axis along which the bars are laid out (`labels`)."""

    y_label: str
    """Axis label for the axis corresponding to bar height (`values`)."""

    labels: Sequence[Union[str, int, float]]
    """Labels for each bar with corresponding height specified in `values`."""

    values: NullableNumberSeries
    """Values for each bar with corresponding label specified in `labels`."""

    config: Optional[AxisConfig] = None
    """Custom format options to allow for control over the display of the numerical plot axis (`values`)."""

    def __post_init_post_parse__(self) -> None:
        n_labels, n_values = len(self.labels), len(self.values)
        if n_labels == 0 or n_values == 0 or n_labels != n_values:
            raise ValueError(
                f"Series 'labels' (length: {n_labels}) and 'values' (length: {n_values}) "
                "must be equal length and non-empty",
            )

    @staticmethod
    def _data_type() -> _PlotType:
        return _PlotType.BAR


@dataclass(frozen=True, config=ValidatorConfig)
class ConfusionMatrix(Plot):
    """
    A confusion matrix. Example:

    ```python
    ConfusionMatrix(
        title="Cat and Dog Confusion",
        labels=["Cat", "Dog"],
        matrix=[[90, 10], [5, 95]],
    )
    ```

    Yields a confusion matrix of the form:

    ```
                Predicted

                Cat   Dog
               +----+----+
           Cat | 90 | 10 |
    Actual     +----+----+
           Dog |  5 | 95 |
               +----+----+
    ```
    """

    title: str
    """The plot title."""

    labels: List[str]
    """The labels corresponding to each entry in the square `matrix`."""

    matrix: Sequence[NullableNumberSeries]
    """A square matrix, typically representing the number of matches between class `i` and class `j`."""

    x_label: str = "Predicted"
    """The label for the `x` axis of the confusion matrix."""

    y_label: str = "Actual"
    """The label for the `y` axis of the confusion matrix."""

    def __post_init_post_parse__(self) -> None:
        n_labels = len(self.labels)
        if n_labels < 2:
            raise ValueError(f"At least two labels required: got {n_labels}")
        if len(self.matrix) != n_labels:
            raise ValueError(f"Invalid number of matrix rows: got {len(self.matrix)}, expected {n_labels}")
        for i, row in enumerate(self.matrix):
            if len(row) != n_labels:
                raise ValueError(f"Invalid number of matrix columns in row {i}: {len(row)}, expected {n_labels}")

    @staticmethod
    def _data_type() -> _PlotType:
        return _PlotType.CONFUSION_MATRIX


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
