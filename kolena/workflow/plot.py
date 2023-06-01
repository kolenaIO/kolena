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
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np
from pydantic.dataclasses import dataclass
from typing_extensions import Literal

from kolena._utils.validators import ValidatorConfig
from kolena.workflow import DataObject
from kolena.workflow._datatypes import DataType
from kolena.workflow._datatypes import TypedDataObject

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
    A plot visualizing distribution of one or more continuous values, e.g. distribution of an error metric across all
    samples within a test case.

    For visualization of discrete values, see :class:`BarPlot`.
    """

    title: str
    x_label: str
    y_label: str

    #: A Histogram requires intervals to bucket the data. For ``n`` buckets, ``n+1`` consecutive bounds must be
    #: specified in increasing order.
    buckets: NumberSeries

    #: For ``n`` buckets, there are ``n`` frequencies corresponding to the height of each bucket. The frequency at index
    #: ``i`` corresponds to the bucket with bounds (``i``, ``i+1``) in ``buckets``.
    #:
    #: To specify multiple distributions for a given test case, multiple frequency series can be provided, corresponding
    #: e.g. to the distribution for a given class within a test case, with name specified in ``labels``.
    frequency: Union[NumberSeries, Sequence[NumberSeries]]

    #: Specify a list of labels corresponding to the different ``frequency`` series when multiple series are provided.
    #: Can be omitted when a single ``frequency`` series is provided.
    labels: Optional[List[str]] = None

    #: Custom format options to allow for control over the display of the plot axes.
    x_config: Optional[AxisConfig] = None
    y_config: Optional[AxisConfig] = None

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
