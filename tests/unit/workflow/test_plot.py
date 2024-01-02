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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Sequence
from typing import Union

import pytest

from kolena.workflow import AxisConfig
from kolena.workflow import BarPlot
from kolena.workflow import ConfusionMatrix
from kolena.workflow import Curve
from kolena.workflow import CurvePlot
from kolena.workflow import Histogram
from kolena.workflow.plot import _PlotType
from kolena.workflow.plot import NullableNumberSeries
from kolena.workflow.plot import NumberSeries


@pytest.mark.parametrize(
    "params",
    [
        dict(x=[1, 2], y=[]),
        dict(x=[1, 2], y=[1]),
        dict(x=[1, 2], y=[1, 2], extra="test"),
        dict(x=[1, 2], y=[1, 2], extra=[1, 2, 3]),
        dict(x=[1, 2], y=[1, 2], extra=dict(a=[])),
        dict(x=[1, 2], y=[1, 2], extra=dict(a=[1])),
        dict(x=[1, 2], y=[1, 2], extra=dict(a=[1, 2, 3])),
        dict(x=[1, 2], y=[1, 2], label="test", extra=dict(a=[1])),
        dict(x=[1, 2], y=[1, 2], label="test", extra=dict(a=None)),
    ],
)
def test__curve__validate__invalid(params: Dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        Curve(**params)


def test__curve_plot__validate() -> None:
    # empty curves is valid configuration
    CurvePlot(title="test", x_label="x", y_label="y", curves=[])

    CurvePlot(
        title="test",
        x_label="x",
        y_label="y",
        curves=[
            Curve(label="a", x=[1, 2, 3], y=[2, 3, 4], extra=dict(a=[1, 2, 3], b=[1, 2, 3])),
            Curve(label="b", x=[1.0, 2.5, 3], y=[-2, -3.0, -4.5], extra=None),
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
        "curves": [{"label": "a", "x": [1, 2, 3], "y": [2, 3, 4], "extra": None}],
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


@pytest.mark.parametrize(
    "buckets,frequency,labels",
    [
        ([1.0, 1.1], [4.2], None),
        ([-3, -2, -1, 0, 1, 2, 3], [0, 1, 2, 1, 4.4, 0.2], None),
        ([1, 2, 3], [[2, 3], [4, 5], [6, 7]], ["a", "b", "c"]),
        ([1, 2, 3], [[2, 3]], ["a"]),
    ],
)
def test__histogram__validate(
    buckets: NumberSeries,
    frequency: Union[NumberSeries, Sequence[NumberSeries]],
    labels: Optional[List[str]],
) -> None:
    Histogram(title="test", x_label="x", y_label="y", buckets=buckets, frequency=frequency, labels=labels)


@pytest.mark.parametrize(
    "buckets,frequency,labels",
    [
        ([], [], None),
        ([1], [], None),
        ([1], [1, 2, 3], None),
        ([1, 2, 3], [1], None),
        (["a", "b"], [90], None),
        ([2, 4, 3.9], [2, 3], None),
        ([-3, -2, -1, 0, 1, 2, -3], [0, 1, 2, 1, 4.4, 0.2], None),
        ([1, 2], [[90]], None),  # labels are required
        ([1, 2, 3], [[1, 2], [3, 4]], ["a"]),
        ([1, 2, 3], [[1, 2], [3, 4]], ["a", "b", "c"]),
        ([1, 2, 3], [[1, 2], [3]], ["a", "b"]),  # jagged
        ([1, 2, 3], [[1, 2], [3, 4, 5]], ["a", "b"]),  # jagged
    ],
)
def test__histogram__validate__invalid(
    buckets: NumberSeries,
    frequency: Union[NumberSeries, Sequence[NumberSeries]],
    labels: Optional[List[str]],
) -> None:
    with pytest.raises(ValueError):
        Histogram(title="test", x_label="x", y_label="y", buckets=buckets, frequency=frequency, labels=labels)


def test__histogram__serialize() -> None:
    assert Histogram(
        title="test",
        x_label="x",
        y_label="y",
        buckets=[-3, -2, -1, 0, 1, 2, 3],
        frequency=[[0, 1, 2, 1, 4.4, 0.2], [0, 1, 2, 3, 4, 5]],
        labels=["a", "b"],
        y_config=AxisConfig(type="log"),
    )._to_dict() == {
        "title": "test",
        "x_label": "x",
        "y_label": "y",
        "buckets": [-3, -2, -1, 0, 1, 2, 3],
        "frequency": [[0, 1, 2, 1, 4.4, 0.2], [0, 1, 2, 3, 4, 5]],
        "labels": ["a", "b"],
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


def test__import__base() -> None:
    from kolena.workflow import AxisConfig  # noqa: F401
    from kolena.workflow import Curve  # noqa: F401
    from kolena.workflow import CurvePlot  # noqa: F401
    from kolena.workflow import Histogram  # noqa: F401
    from kolena.workflow import BarPlot  # noqa: F401
    from kolena.workflow import ConfusionMatrix  # noqa: F401


def test__import__module() -> None:
    from kolena.workflow.plot import AxisConfig  # noqa: F401
    from kolena.workflow.plot import Plot  # noqa: F401
    from kolena.workflow.plot import Curve  # noqa: F401
    from kolena.workflow.plot import CurvePlot  # noqa: F401
    from kolena.workflow.plot import Histogram  # noqa: F401
    from kolena.workflow.plot import BarPlot  # noqa: F401
    from kolena.workflow.plot import ConfusionMatrix  # noqa: F401


def test__import__backcompat() -> None:
    from kolena.workflow.evaluator import AxisConfig  # noqa: F401
    from kolena.workflow.evaluator import Plot  # noqa: F401
    from kolena.workflow.evaluator import Curve  # noqa: F401
    from kolena.workflow.evaluator import CurvePlot  # noqa: F401
    from kolena.workflow.evaluator import Histogram  # noqa: F401
    from kolena.workflow.evaluator import BarPlot  # noqa: F401
    from kolena.workflow.evaluator import ConfusionMatrix  # noqa: F401
