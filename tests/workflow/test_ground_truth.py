from dataclasses import dataclass
from typing import List
from typing import Optional
from typing import Type
from typing import Union

import pytest

from .data import ComplexBoundingBox
from .data import ImageTriplet
from .data import NestedComplexBoundingBox
from kolena.workflow import GroundTruth
from kolena.workflow import TestSample
from kolena.workflow._datatypes import DataObject
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.annotation import Keypoints
from kolena.workflow.annotation import LabeledBoundingBox
from kolena.workflow.annotation import LabeledPolygon
from kolena.workflow.annotation import Polygon
from kolena.workflow.annotation import Polyline
from kolena.workflow.asset import ImageAsset
from kolena.workflow.ground_truth import _validate_ground_truth_type


def test__validate(test_sample_type: Type[TestSample]) -> None:
    @dataclass(frozen=True)
    class Tester(GroundTruth):
        a: str
        b: bool
        c: int
        d: float
        e: Optional[int]
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

    _validate_ground_truth_type(test_sample_type, Tester)


def test__validate__composite() -> None:
    @dataclass(frozen=True)
    class FaceRegion(DataObject):
        a: int
        b: float
        c: BoundingBox
        d: Keypoints

    @dataclass(frozen=True)
    class Tester(GroundTruth):
        a: FaceRegion
        b: FaceRegion
        x: int
        y: List[BoundingBox]

    @dataclass(frozen=True)
    class TesterAlt(GroundTruth):
        a: FaceRegion
        b: List[BoundingBox]
        x: int

    _validate_ground_truth_type(ImageTriplet, Tester)
    _validate_ground_truth_type(ImageTriplet, TesterAlt)


def test__validate__composite_invalid() -> None:
    @dataclass(frozen=True)
    class Tester(GroundTruth):
        a: ComplexBoundingBox
        y: ComplexBoundingBox
        b: List[BoundingBox]

    with pytest.raises(ValueError):
        _validate_ground_truth_type(ImageTriplet, Tester)


def test__validate__nested_composite_invalid() -> None:
    @dataclass(frozen=True)
    class Tester(GroundTruth):
        a: NestedComplexBoundingBox
        b: List[BoundingBox]

    with pytest.raises(ValueError):
        _validate_ground_truth_type(ImageTriplet, Tester)


def test__validate__list_of_composite_invalid() -> None:
    @dataclass(frozen=True)
    class Tester(GroundTruth):
        a: List[ComplexBoundingBox]
        b: List[BoundingBox]

    with pytest.raises(ValueError):
        _validate_ground_truth_type(ImageTriplet, Tester)
