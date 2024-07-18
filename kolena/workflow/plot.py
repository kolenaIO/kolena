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
This module surfaces plot definitions to visualize test-case-level data. [Evaluator](evaluator.md) implementations can
optionally compute plots using these definitions for visualization on the
[:kolena-results-16: Results](https://app.kolena.com/redirect/results) page.

The following plot types are available:

- [`CurvePlot`][kolena.workflow.CurvePlot]
- [`Histogram`][kolena.workflow.Histogram]
- [`BarPlot`][kolena.workflow.BarPlot]
- [`ConfusionMatrix`][kolena.workflow.ConfusionMatrix]
"""
from abc import ABCMeta
from typing import Dict
from typing import List
from typing import Literal
from typing import Optional
from typing import Sequence
from typing import Union

import numpy as np

from kolena._utils.datatypes import DataCategory
from kolena._utils.datatypes import DataObject
from kolena._utils.datatypes import DataType
from kolena._utils.datatypes import TypedDataObject
from kolena._utils.pydantic_v1.dataclasses import dataclass
from kolena._utils.validators import ValidatorConfig

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
    def _data_category() -> DataCategory:
        return DataCategory.PLOT


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

    extra: Optional[Dict[str, NumberSeries]] = None
    """
    Optionally specify additional series shown when hovering over the plot. For example, when plotting a
    precision-recall curve, it is desirable to include an extra series `threshold` to specify the confidence threshold
    value at which a given precision-recall point occurs.
    """

    def __post_init_post_parse__(self) -> None:
        if len(self.x) != len(self.y):
            raise ValueError(
                f"Series 'x' (length: {len(self.x)}) and 'y' (length: {len(self.y)}) have different lengths",
            )
        for key, series in (self.extra or {}).items():
            if len(series) != len(self.x):
                raise ValueError(
                    f"Series '{key}' (length: {len(series)}) must match length of 'x' and 'y' (length: {len(self.x)})",
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
