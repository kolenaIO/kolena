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
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import pytest

from kolena.workflow._datatypes import DataObject
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


def test__validate__metrics_test_case__valid_nested() -> None:
    @dataclasses.dataclass(frozen=True)
    class Nested(MetricsTestCase):
        a: int
        b: str
        c: float
        d: Optional[str]
        e: Union[int, str, float]

    @dataclasses.dataclass(frozen=True)
    class NestedTester(MetricsTestCase):
        a: int
        b: float
        c: List[Nested]
        d: List[Nested]


def test__validate__metrics_test_case__invalid_nested() -> None:
    @dataclasses.dataclass(frozen=True)
    class Nested(MetricsTestCase):
        a: int

    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class NestedTester(MetricsTestCase):
            a: Nested  # only List[MetricsTestCase] is allowed

    with pytest.raises(ValueError):

        @dataclasses.dataclass(frozen=True)
        class Nested2DListTester(MetricsTestCase):
            a: List[List[Nested]]  # only single List[MetricsTestCase] is allowed
