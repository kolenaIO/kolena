from dataclasses import dataclass
from typing import Optional

from kolena.workflow import Composite
from kolena.workflow import DataObject
from kolena.workflow import Image
from kolena.workflow import Metadata
from kolena.workflow.annotation import BoundingBox
from kolena.workflow.asset import ImageAsset


class ImageTriplet(Composite):
    a: Image
    b: Image
    c: Image
    d: str
    e: Optional[ImageAsset]
    metadata: Metadata


@dataclass(frozen=True)
class ComplexBoundingBox(DataObject):
    a: int
    b: float
    c: BoundingBox


@dataclass(frozen=True)
class NestedComplexBoundingBox(DataObject):
    a: ComplexBoundingBox
    b: int
