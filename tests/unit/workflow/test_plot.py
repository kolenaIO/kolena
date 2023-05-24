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
from typing import Any
from typing import List
from typing import Optional
from typing import Union

import pytest

from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow.evaluator import _PlotType
from kolena.workflow.evaluator import AxisConfig
from kolena.workflow.evaluator import BarPlot
from kolena.workflow.evaluator import ConfusionMatrix
from kolena.workflow.evaluator import Histogram
from kolena.workflow.evaluator import HistogramDistribution
from kolena.workflow.evaluator import NullableNumberSeries


def test__curve_plot__validate() -> None:
    # empty curves is valid configuration
    CurvePlot(title="test", x_label="x", y_label="y", curves=[])

    CurvePlot(
        title="test",
        x_label="x",
        y_label="y",
        curves=[
            Curve(label="a", x=[1, 2, 3], y=[2, 3, 4]),
            Curve(label="b", x=[1.0, 2.5, 3], y=[-2, -3.0, -4.5]),
        ],
    )


def test__curve_plot__validate__invalid() -> None:
    # different length
    with pytest.raises(ValueError):
        CurvePlot(title="test", x_label="x", y_label="y", curves=[Curve(label="a", x=[1, 2], y=[1, 2, 3])])

    # invalid values
    with pytest.raises(ValueError):
        CurvePlot(title="test", x_label="x", y_label="y", curves=[Curve(label="a", x=[1, None], y=[1, 2])])

    # invalid axis type
    with pytest.raises(ValueError):
        CurvePlot(
            title="test",
            x_label="x",
            y_label="y",
            curves=[Curve(label="a", x=[1, 2, 3], y=[2, 3, 4])],
            x_config=AxisConfig(type="other"),
        )


def test__curve_plot__serialize() -> None:
    assert CurvePlot(
        title="test",
        x_label="x",
        y_label="y",
        curves=[Curve(label="a", x=[1, 2, 3], y=[2, 3, 4])],
        y_config=AxisConfig(type="log"),
    )._to_dict() == {
        "title": "test",
        "x_label": "x",
        "y_label": "y",
        "curves": [{"label": "a", "x": [1, 2, 3], "y": [2, 3, 4]}],
        "data_type": f"{_PlotType._data_category()}/{_PlotType.CURVE.value}",
        "x_config": None,
        "y_config": {"type": "log"},
    }


def test__confusion_matrix__validate() -> None:
    ConfusionMatrix(
        title="Cat and Dog Confusion",
        x_label="Predicted",
        y_label="Actual",
        labels=["Cat", "Dog"],
        matrix=[[90, 10], [5, 95]],
    )


@pytest.mark.parametrize(
    "labels,matrix",
    [
        ([], []),
        ([], [[]]),
        (["a"], [[90]]),
        (["a", "b"], [90, 10]),
        (["a", "b"], [90, 10, 80, 20]),
        (["a", "b"], [[90, 10], [80]]),
        (["a", "b"], [[90, 10], [80, 20, 10]]),
        (["a", "b"], [[90, 10, 0], [80, 20]]),
        (["a", "b"], [[90, 10], [80, 20], [10, 10]]),
        (["a", "b", "c"], [[90, 10], [80, 20]]),
        (["a", "b", "c"], [[90, 10], [80, 20], [10, 10]]),
    ],
)
def test__confusion_matrix__validate__invalid(labels: List[str], matrix: List[Any]) -> None:
    with pytest.raises(ValueError):
        ConfusionMatrix(title="t", x_label="x", y_label="a", labels=labels, matrix=matrix)


def test__confusion_matrix__serialize() -> None:
    confusion_matrix = ConfusionMatrix(
        title="t",
        x_label="x",
        y_label="y",
        labels=["a", "b"],
        matrix=[[90, 10], [20, 80]],
    )

    assert confusion_matrix._to_dict() == dict(
        title="t",
        x_label="x",
        y_label="y",
        labels=["a", "b"],
        matrix=[[90, 10], [20, 80]],
        data_type=f"{_PlotType._data_category()}/{_PlotType.CONFUSION_MATRIX.value}",
    )

    # TODO: list of lists fails to deserialize, but not critical for plots (plots are never pulled down)
    # assert ConfusionMatrix._from_dict(confusion_matrix._to_dict()) == confusion_matrix


def test__histogram__validate() -> None:
    Histogram(title="mini", x_label="x", y_label="y", buckets=[1.0, 1.1], frequency=[4.2])

    Histogram(
        title="test",
        x_label="x",
        y_label="y",
        buckets=[-3, -2, -1, 0, 1, 2, 3],
        frequency=[0, 1, 2, 1, 4.4, 0.2],
    )


@pytest.mark.parametrize(
    "buckets,frequency,distributions",
    [
        ([], [], None),
        ([1], [], None),
        ([1], [1, 2, 3], None),
        ([1, 2, 3], [1], None),
        ([1, 2], [[90]], None),
        (["a", "b"], [90], None),
        ([2, 4, 3.9], [2, 3], None),
        ([-3, -2, -1, 0, 1, 2, -3], [0, 1, 2, 1, 4.4, 0.2], None),
        ([0, 1, 2], [0, 1], [HistogramDistribution(buckets=[0, 1], frequency=[1], label="a")]),
    ],
)
def test__histogram__validate__invalid(
    buckets: List[Any],
    frequency: List[Any],
    distributions: Optional[List[HistogramDistribution]],
) -> None:
    # different length
    with pytest.raises(ValueError):
        Histogram(
            title="test",
            x_label="x",
            y_label="y",
            buckets=buckets,
            frequency=frequency,
            distributions=distributions,
        )


@pytest.mark.parametrize(
    "buckets,frequency",
    [
        ([], []),
        ([1], []),
        ([1], [1, 2, 3]),
        ([1, 2, 3], [1]),
        ([1, 2], [[90]]),
        (["a", "b"], [90]),
        ([2, 4, 3.9], [2, 3]),
        ([-3, -2, -1, 0, 1, 2, -3], [0, 1, 2, 1, 4.4, 0.2]),
    ],
)
def test__histogram_distribution__validate__invalid(buckets: List[Any], frequency: List[Any]) -> None:
    with pytest.raises(ValueError):
        HistogramDistribution(label="a", buckets=buckets, frequency=frequency)


def test__histogram__serialize() -> None:
    assert Histogram(
        title="test",
        x_label="x",
        y_label="y",
        buckets=[-3, -2, -1, 0, 1, 2, 3],
        frequency=[0, 1, 2, 1, 4.4, 0.2],
        y_config=AxisConfig(type="log"),
    )._to_dict() == {
        "title": "test",
        "x_label": "x",
        "y_label": "y",
        "buckets": [-3, -2, -1, 0, 1, 2, 3],
        "frequency": [0, 1, 2, 1, 4.4, 0.2],
        "distributions": [],
        "x_config": None,
        "y_config": {"type": "log"},
        "data_type": f"{_PlotType._data_category()}/{_PlotType.HISTOGRAM.value}",
    }


@pytest.mark.parametrize(
    "valid,labels,values",
    [
        (True, ["a", "b"], [1, 2]),
        (True, ["a"], [1]),
        (True, list(str(i) for i in range(10)), list(range(10))),
        (True, ["a", "b"], [1.5, None]),
        (True, [1, 2], [1.5, None]),
        (True, ["a", 2, 3.5], [1.5, None, -1]),
        (False, ["a"], [1, 2, 3]),
        (False, ["a", "b"], [1]),
        (False, ["a"], []),
        (False, [], [1]),
        (False, [], []),
    ],
)
def test__bar_plot__validate(valid: bool, labels: List[Union[str, int, float]], values: NullableNumberSeries) -> None:
    if valid:
        BarPlot(title="tester", x_label="x", y_label="y", labels=labels, values=values)
    else:
        with pytest.raises(ValueError):
            BarPlot(title="tester", x_label="x", y_label="y", labels=labels, values=values)
