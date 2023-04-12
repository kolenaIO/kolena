from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

import pytest

from .data import ComplexBoundingBox
from .data import ImageTriplet
from .data import NestedComplexBoundingBox
from kolena.workflow import Inference
from kolena.workflow import TestSample
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import Polygon
from kolena.workflow.annotation import Polyline
from kolena.workflow.asset import ImageAsset
from kolena.workflow.inference import _validate_inference_type


def test__validate() -> None:
    @dataclass(frozen=True)
    class Tester(Inference):
        a: str
        b: bool
        c: int
        d: float
        e: Optional[str]
        f: BoundingBox
        g: LabeledBoundingBox
        h: Polygon
        i: LabeledPolygon
        j: Keypoints
        l: Polyline
        m: ImageAsset
        n: List[int]
        o: List[Polygon]
        p: Optional[str]
        q: Optional[LabeledPolygon]
        r: Union[float, BoundingBox, ImageAsset]

    _validate_inference_type(TestSample, Tester)


def test__validate__invalid() -> None:
    @dataclass(frozen=True)
    class Tester(Inference):
        a: Dict[str, Any]

    with pytest.raises(ValueError):
        _validate_inference_type(TestSample, Tester)


def test__validate__composite() -> None:
    @dataclass(frozen=True)
    class Tester(Inference):
        a: ComplexBoundingBox
        b: ComplexBoundingBox
        x: int
        y: List[BoundingBox]

    @dataclass(frozen=True)
    class TesterAlt(Inference):
        a: ComplexBoundingBox
        b: List[BoundingBox]
        x: int

    _validate_inference_type(ImageTriplet, Tester)
    _validate_inference_type(ImageTriplet, TesterAlt)


def test__validate__composite_invalid() -> None:
    @dataclass(frozen=True)
    class Tester(Inference):
        a: ComplexBoundingBox
        y: ComplexBoundingBox
        b: List[BoundingBox]

    with pytest.raises(ValueError):
        _validate_inference_type(ImageTriplet, Tester)


def test__validate__nested_composite_invalid() -> None:
    @dataclass(frozen=True)
    class Tester(Inference):
        a: NestedComplexBoundingBox
        b: List[BoundingBox]

    with pytest.raises(ValueError):
        _validate_inference_type(ImageTriplet, Tester)


def test__validate__list_of_composite_invalid() -> None:
    @dataclass(frozen=True)
    class Tester(Inference):
        a: List[ComplexBoundingBox]
        b: List[BoundingBox]

    with pytest.raises(ValueError):
        _validate_inference_type(ImageTriplet, Tester)
