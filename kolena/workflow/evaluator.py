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

from pydantic import validate_arguments
from pydantic.dataclasses import dataclass
from pydantic.typing import Literal

from kolena._api.v1.generic import TestRun as API
from kolena._utils.validators import ValidatorConfig
from kolena.workflow import GroundTruth
from kolena.workflow import Inference
from kolena.workflow import TestCase
from kolena.workflow import TestSample
from kolena.workflow import TestSuite
from kolena.workflow._datatypes import DataObject
from kolena.workflow._datatypes import DataType
from kolena.workflow._datatypes import TypedDataObject
from kolena.workflow._validators import validate_data_object_type
from kolena.workflow._validators import validate_scalar_data_object_type


@dataclass(frozen=True, config=ValidatorConfig)
class MetricsTestSample(DataObject, metaclass=ABCMeta):
    """
    Test-sample-level metrics produced by an :class:`Evaluator`.

    This class should be subclassed with the relevant fields for a given workflow.

    Examples here may include the number of true positive detections on an image, the mean IOU of inferred polygon(s)
    with ground truth polygon(s), etc.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _validate_metrics_test_sample_type(cls)


@dataclass(frozen=True, config=ValidatorConfig)
class MetricsTestCase(DataObject, metaclass=ABCMeta):
    """
    Test-case-level metrics produced by an :class:`Evaluator`.

    This class should be subclassed with the relevant fields for a given workflow.

    Test-case-level metrics are aggregate metrics like Precision, Recall, and F1 score. Any and all aggregate metrics
    that fit a workflow should be defined here.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _validate_metrics_test_case_type(cls)


@dataclass(frozen=True, config=ValidatorConfig)
class MetricsTestSuite(DataObject, metaclass=ABCMeta):
    """
    Test-suite-level metrics produced by an :class:`Evaluator`.

    This class should be subclassed with the relevant fields for a given workflow.

    Test-suite-level metrics typically measure performance across test cases, e.g. penalizing variance across different
    subsets of a benchmark.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        _validate_metrics_test_suite_type(cls)


NumberSeries = Sequence[Union[float, int]]
NullableNumberSeries = Sequence[Union[float, int, None]]


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
    """Configuration for the format of a given axis on a Plot"""

    #: Type of axis to display. Supported options are `linear` and `log`.
    type: Literal["linear", "log"]


@dataclass(frozen=True, config=ValidatorConfig)
class Plot(TypedDataObject[_PlotType], metaclass=ABCMeta):
    """A data visualization shown when exploring model results in the web platform."""


@dataclass(frozen=True, config=ValidatorConfig)
class Curve(DataObject):
    """A single series on a :class:`CurvePlot`."""

    x: NumberSeries
    y: NumberSeries

    #: Optionally specify an additional label (in addition to the associated test case) to apply to this curve, for use
    #: when e.g. there are multiple curves generated per test case.
    label: Optional[str] = None

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
    x_label: str
    y_label: str

    #: A test case may generate zero or more curves on a given plot. However, under most circumstances, a single curve
    #: per test case is desirable.
    curves: List[Curve]

    #: Custom format options to allow for control over the display of the plot axes.
    x_config: Optional[AxisConfig] = None
    y_config: Optional[AxisConfig] = None

    @staticmethod
    def _data_type() -> _PlotType:
        return _PlotType.CURVE


@dataclass(frozen=True, config=ValidatorConfig)
class Histogram(Plot):
    """
    A plot visualizing a histogram per test case.

    Histograms allow for easy visualizations of data distributions.
    """

    title: str
    x_label: str
    y_label: str

    #: A Histogram requires intervals to bucket the data.
    #: For n buckets, n+1 consecutive bounds must be specified in increasing order.
    buckets: NumberSeries

    #: For n buckets, there are n frequencies that define each bucket's height.
    #: The nth frequency corresponds to the nth bucket.
    frequency: NumberSeries

    #: Custom format options to allow for control over the display of the plot axes.
    x_config: Optional[AxisConfig] = None
    y_config: Optional[AxisConfig] = None

    def __post_init_post_parse__(self) -> None:
        if len(self.frequency) + 1 != len(self.buckets):
            long_err_msg = (
                f"Series 'frequency' (length: {len(self.frequency)}) "
                f"and 'buckets' (length: {len(self.buckets)}) should be 1 apart"
            )
            raise ValueError(long_err_msg)

        for i in range(len(self.buckets) - 1):
            if self.buckets[i] >= self.buckets[i + 1]:
                raise ValueError(
                    f"At index {i}, {i + 1}, series 'buckets' is ({self.buckets[i]}, {self.buckets[i + 1]})",
                )

    @staticmethod
    def _data_type() -> _PlotType:
        return _PlotType.HISTOGRAM


@dataclass(frozen=True, config=ValidatorConfig)
class BarPlot(Plot):
    """A plot visualizing a set of bars per test case."""

    title: str

    #: Axis label for the axis along which the bars are laid out (``labels``).
    x_label: str

    #: Axis label for the axis corresponding to bar height (``values``).
    y_label: str

    #: Labels for each bar with corresponding height specified in ``values``.
    labels: Sequence[Union[str, int, float]]

    #: Values for each bar with corresponding label specified in ``labels``.
    values: NullableNumberSeries

    #: Custom format options to allow for control over the display of the numerical plot axis (``values``).
    config: Optional[AxisConfig] = None

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

    .. code-block:: python

        ConfusionMatrix(
            title="Cat and Dog Confusion",
            labels=["Cat", "Dog"],
            matrix=[[90, 10], [5, 95]],
        )

    Yields a confusion matrix of the form:

    .. code-block:: none

                    Predicted

                    Cat   Dog
                   +----+----+
               Cat | 90 | 10 |
        Actual     +----+----+
               Dog |  5 | 95 |
                   +----+----+
    """

    title: str
    labels: List[str]
    matrix: Sequence[NullableNumberSeries]
    x_label: str = "Predicted"
    y_label: str = "Actual"

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
    Configuration for an :class:`Evaluator`.

    Example evaluator configurations may specify:

    - Fixed confidence thresholds at which detections are discarded.
    - Different algorithms/strategies used to compute confidence thresholds
        (e.g. "accuracy optimal" for a classification-type workflow).
    """

    @abstractmethod
    def display_name(self) -> str:
        raise NotImplementedError


class Evaluator(metaclass=ABCMeta):
    """
    An :class:`kolena.workflow.Evaluator` transforms inferences into metrics.

    Metrics are computed at the individual test sample level (:class:`kolena.workflow.MetricsTestSample`), in aggregate
    at the test case level (:class:`kolena.workflow.MetricsTestCase`), and across populations at the test suite level
    (:class:`kolena.workflow.MetricsTestSuite`).

    Test-case-level plots (:class:`kolena.workflow.Plot`) may also be computed.
    """

    configurations: List[EvaluatorConfiguration]

    @validate_arguments(config=ValidatorConfig)
    def __init__(self, configurations: Optional[List[EvaluatorConfiguration]] = None):
        if configurations is not None and len(configurations) == 0:
            raise ValueError("empty configuration list provided, at least one configuration required or 'None'")
        self.configurations = configurations or []
        if len({configuration.display_name() for configuration in self.configurations}) < len(self.configurations):
            raise ValueError("all configurations must have distinct display names")

    def display_name(self) -> str:
        return type(self).__name__

    @abstractmethod
    def compute_test_sample_metrics(
        self,
        test_case: TestCase,
        inferences: List[Tuple[TestSample, GroundTruth, Inference]],
        configuration: Optional[EvaluatorConfiguration] = None,
    ) -> List[Tuple[TestSample, MetricsTestSample]]:
        """
        Compute metrics for every test sample in a test case.

        Must be implemented.

        :param test_case: the test case to which the provided test samples and ground truths belong.
        :param inferences: the test samples, ground truths, and inferences for all entries in a test case.
        :param configuration: the evaluator configuration to use. Empty for implementations that are not configured.
        :return: test-sample-level metrics for each provided test sample.
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
        Compute aggregate metrics across a test case.

        Must be implemented.

        :param test_case: the test case in question.
        :param inferences: the test samples, ground truths, and inferences for all entries in a test case.
        :param metrics: the test-sample-level metrics computed by :meth:`Evaluator.compute_test_sample_metrics`.
            Provided in the same order as ``inferences``.
        :param configuration: the evaluator configuration to use. Empty for implementations that are not configured.
        :return: test-case-level metrics for the provided test case.
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

        :param test_case: the test case in question
        :param inferences: the test samples, ground truths, and inferences for all entries in a test case.
        :param metrics: the test-sample-level metrics computed by :meth:`Evaluator.compute_test_sample_metrics`.
            Provided in the same order as ``inferences``.
        :param configuration: the evaluator configuration to use. Empty for implementations that are not configured.
        :return: zero or more plots for this test case at this configuration.
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
        Optionally compute test-suite-level metrics.

        :param test_suite: the test suite in question
        :param metrics: the test-case-level metrics computed by :meth:`Evaluator.compute_test_case_metrics`
        :param configuration: the evaluator configuration to use. Empty for implementations that are not configured.
        :return: the test-suite-level metrics for this test suite
        """
        return None  # not required


def _validate_metrics_test_sample_type(metrics_test_sample_type: Type[MetricsTestSample]) -> None:
    # TODO: support special structure for ImagePair test sample types?
    validate_data_object_type(metrics_test_sample_type)


def _validate_metrics_test_case_type(metrics_test_case_type: Type[DataObject]) -> None:
    validate_scalar_data_object_type(metrics_test_case_type)


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
