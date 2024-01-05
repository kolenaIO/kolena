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
import dataclasses
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import pydantic
import pytest

from kolena._experimental.workflow.thresholded import ThresholdedMetrics
from kolena._utils.datatypes import DataObject
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Polyline
from kolena.workflow.evaluator import MetricsTestCase
from kolena.workflow.evaluator import MetricsTestSample
from kolena.workflow.evaluator import MetricsTestSuite


def test__validate__metrics_test_sample() -> None:
    @dataclasses.dataclass(frozen=True)
    class Tester(MetricsTestSample):
        a: float
        b: Optional[str]
        c: Union[bool, int]
        d: List[float]
        e: BoundingBox


def test__validate__metrics_test_sample__invalid__bytes() -> None:
    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class BytesTester(MetricsTestSample):
            a: Optional[bytes]


def test__validate__metrics_test_sample__invalid__dict() -> None:
    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class DictTester(MetricsTestSample):
            a: Dict[str, Any]


def test__validate__metrics_test_sample__invalid__nested() -> None:
    @dataclasses.dataclass(frozen=True)
    class Nested(DataObject):
        a: float

    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class NestedTester(MetricsTestSample):
            a: Nested


@pytest.mark.skip("special handling for image pair test sample metrics is not currently supported")
def test__validate__metrics_test_sample__image_pair() -> None:
    @dataclasses.dataclass(frozen=True)
    class Inner(MetricsTestSample):
        a: float
        b: BoundingBox

    @dataclasses.dataclass(frozen=True)
    class Inner2(MetricsTestSample):
        a: Polyline

    @dataclasses.dataclass(frozen=True)
    class Tester(MetricsTestSample):  # does not throw
        a: Inner
        b: Inner2
        c: Union[bool, int]


def test__validate__metrics_test_sample__image_pair__invalid() -> None:
    @dataclasses.dataclass(frozen=True)
    class InnerInner(DataObject):
        a: Polyline

    @dataclasses.dataclass(frozen=True)
    class Inner(DataObject):
        a: InnerInner

    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class Tester(MetricsTestSample):
            a: Inner


@pytest.mark.parametrize("base", [MetricsTestCase, MetricsTestSuite])
def test__validate__metrics_test_case_and_test_suite(base: Type[DataObject]) -> None:
    @dataclasses.dataclass(frozen=True)
    class EmptyTester(base):  # does not throw on __init_subclass__
        ...

    @dataclasses.dataclass(frozen=True)
    class ScalarTester(base):
        a: int
        b: float
        c: str
        d: bool

    @dataclasses.dataclass(frozen=True)
    class OptionalTester(base):
        a: Optional[int]

    @dataclasses.dataclass(frozen=True)
    class UnionTester(base):
        a: Union[int, float, str]


@pytest.mark.parametrize("base", [MetricsTestCase, MetricsTestSuite])
def test__validate__metrics_test_case_and_test_suite__invalid(base: Type[DataObject]) -> None:
    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class BytesTester(base):
            a: bytes

    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class StructuredTester(base):
            a: BoundingBox
            b: Polyline

    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class DictTester(base):
            a: Dict[str, Any]

    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class ListTester(base):
            a: List[float]


def test__validate__metrics_test_suite__invalid_nested() -> None:
    @dataclasses.dataclass(frozen=True)
    class Nested(DataObject):
        a: int

    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class NestedTester(MetricsTestSuite):
            a: Nested


@dataclasses.dataclass(frozen=True)
class Nested(MetricsTestCase):
    a: int
    b: str
    c: float
    d: Optional[str]
    e: Union[int, str, float]


def test__validate__metrics_test_case__valid_nested() -> None:
    @dataclasses.dataclass(frozen=True)
    class Nested2(MetricsTestCase):
        a: int

    @dataclasses.dataclass(frozen=True)
    class NestedTester(MetricsTestCase):
        a: int
        b: float
        c: List[Nested]
        d: Optional[List[Nested]]
        e: Union[List[Nested], List[Nested2]]
        f: List[Union[Nested, Nested2]]


def test__validate__metrics_test_case__invalid_nested__single() -> None:
    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class NestedTester(MetricsTestCase):
            a: Nested  # only List[MetricsTestCase] is allowed


def test__validate__metrics_test_case__invalid_nested__data_object() -> None:
    @dataclasses.dataclass(frozen=True)
    class Nested(DataObject):  # note base class
        a: int

    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class NestedTester(MetricsTestCase):
            a: List[Nested]  # only List[MetricsTestCase] is allowed


def test__validate__metrics_test_case__invalid_nested__optional() -> None:
    with pytest.raises(ValueError):

        @pydantic.dataclasses.dataclass(frozen=True)
        class Nested2DListTester(MetricsTestCase):
            a: Optional[Nested]  # only single List[MetricsTestCase] is allowed


def test__validate__metrics_test_case__invalid_nested__list_list() -> None:
    with pytest.raises(ValueError):

        @pydantic.dataclasses.dataclass(frozen=True)
        class Nested2DListTester(MetricsTestCase):
            a: List[List[Nested]]  # only single List[MetricsTestCase] is allowed


def test__validate__metrics_test_case__invalid_nested__list_optional() -> None:
    with pytest.raises(ValueError):

        @pydantic.dataclasses.dataclass(frozen=True)
        class Nested2DListTester(MetricsTestCase):
            a: List[Optional[Nested]]


def test__validate__metrics_test_case__invalid_nested__doubly_nested() -> None:
    @dataclasses.dataclass(frozen=True)
    class NestedNested(MetricsTestCase):
        a: List[Nested]

    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class Nested2DTester(MetricsTestCase):
            a: List[NestedNested]  # only one layer of nesting allowed


def test__validate__metrics_test_case__fail_overwrite_field() -> None:
    with pytest.raises(TypeError):

        @dataclasses.dataclass(frozen=True)
        class MyThresholdedMetrics(ThresholdedMetrics):
            threshold: str  # overwrite type


def test__validate__metrics_test_sample__with_thresholded_metric_field() -> None:
    @dataclasses.dataclass(frozen=True)
    class MyThresholded(ThresholdedMetrics):
        score: float

    @dataclasses.dataclass(frozen=True)
    class MyMetrics(MetricsTestSample):
        thresholded_scores: List[MyThresholded]

    sample = MyMetrics(
        thresholded_scores=[
            MyThresholded(threshold=0.1, score=0.5),
            MyThresholded(threshold=0.5, score=0.6),
            MyThresholded(threshold=0.9, score=0.7),
        ],
    )

    assert len(sample.thresholded_scores) == 3
    assert sample.thresholded_scores[0].threshold == 0.1
    assert sample.thresholded_scores[0].score == 0.5
    assert sample.thresholded_scores[1].threshold == 0.5
    assert sample.thresholded_scores[1].score == 0.6
    assert sample.thresholded_scores[2].threshold == 0.9
    assert sample.thresholded_scores[2].score == 0.7
