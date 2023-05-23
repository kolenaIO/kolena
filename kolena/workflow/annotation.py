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
from typing import Dict
from typing import List
from typing import Tuple

from pydantic.dataclasses import dataclass

from kolena._utils.validators import ValidatorConfig
from kolena.workflow._datatypes import DataType
from kolena.workflow._datatypes import TypedDataObject


class _AnnotationType(DataType):
    BOUNDING_BOX = "BOUNDING_BOX"
    POLYGON = "POLYGON"
    POLYLINE = "POLYLINE"
    KEYPOINTS = "KEYPOINTS"
    BOUNDING_BOX_3D = "BOUNDING_BOX_3D"
    SEGMENTATION_MASK = "SEGMENTATION_MASK"
    BITMAP_MASK = "BITMAP_MASK"
    CLASSIFICATON_LABEL = "LABEL"

    @staticmethod
    def _data_category() -> str:
        return "ANNOTATION"


@dataclass(frozen=True, config=ValidatorConfig)
class Annotation(TypedDataObject[_AnnotationType], metaclass=ABCMeta):
    """
    Where applicable, annotations are visualized in the web platform.

    For example, when viewing images, any annotations present in a :class:`kolena.workflow.TestSample`,
    :class:`kolena.workflow.GroundTruth`, :class:`kolena.workflow.Inference`, or
    :class:`kolena.workflow.MetricsTestSample` are rendered on top of the image.
    """


@dataclass(frozen=True, config=ValidatorConfig)
class BoundingBox(Annotation):
    """Rectangular bounding box specified with pixel coordinates of the top left and bottom right vertices."""

    top_left: Tuple[float, float]
    bottom_right: Tuple[float, float]

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.BOUNDING_BOX


@dataclass(frozen=True, config=ValidatorConfig)
class LabeledBoundingBox(BoundingBox):
    """
    Rectangular bounding box specified with pixel coordinates of the top left and bottom right vertices and a string
    label.
    """

    label: str


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredBoundingBox(BoundingBox):
    """
    Rectangular bounding box specified with pixel coordinates of the top left and bottom right vertices and a float
    score.
    """

    score: float


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredLabeledBoundingBox(BoundingBox):
    """
    Rectangular bounding box specified with pixel coordinates of the top left and bottom right vertices, a string
    label, and a float score.
    """

    label: str
    score: float


@dataclass(frozen=True, config=ValidatorConfig)
class Polygon(Annotation):
    """Arbitrary polygon specified by three or more pixel coordinates."""

    points: List[Tuple[float, float]]

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.POLYGON

    def __post_init__(self) -> None:
        if len(self.points) < 3:
            raise ValueError(f"{type(self).__name__} must have at least three points ({len(self.points)} provided)")


@dataclass(frozen=True, config=ValidatorConfig)
class LabeledPolygon(Polygon):
    """Arbitrary polygon specified by three or more pixel coordinates and a string label."""

    label: str


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredPolygon(Polygon):
    """
    Arbitrary polygon specified by three or more pixel coordinates and a float score.
    """

    score: float


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredLabeledPolygon(Polygon):
    """
    Arbitrary polygon specified by three or more pixel coordinates with a string label and a float score.
    """

    label: str
    score: float


@dataclass(frozen=True, config=ValidatorConfig)
class Keypoints(Annotation):
    """Array of any number of keypoints specified in pixel coordinates."""

    points: List[Tuple[float, float]]

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.KEYPOINTS


@dataclass(frozen=True, config=ValidatorConfig)
class Polyline(Annotation):
    """Polyline with any number of vertices specified in pixel coordinates."""

    points: List[Tuple[float, float]]

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.POLYLINE


@dataclass(frozen=True, config=ValidatorConfig)
class BoundingBox3D(Annotation):
    """
    Three-dimensional cuboid bounding box in a right-handed coordinate system.

    Specified by (x, y, z) coordinates for the ``center`` of the cuboid, (x, y, z) ``dimensions``, and a ``rotation``
    parameter specifying the degrees of rotation about each axis (x, y, z) ranging [-pi, pi].
    """

    #: (x, y, z) coordinates specifying the center of the bounding box.
    center: Tuple[float, float, float]

    #: (x, y, z) measurements specifying the dimensions of the bounding box.
    dimensions: Tuple[float, float, float]

    #: Rotations in degrees about each (x, y, z) axis.
    rotations: Tuple[float, float, float]

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.BOUNDING_BOX_3D


@dataclass(frozen=True, config=ValidatorConfig)
class LabeledBoundingBox3D(BoundingBox3D):
    """:class:`BoundingBox3D` with an additional string label."""

    label: str


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredBoundingBox3D(BoundingBox3D):
    """:class:`BoundingBox3D` with an additional float score."""

    score: float


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredLabeledBoundingBox3D(BoundingBox3D):
    """:class:`BoundingBox3D` with an additional string label and float score."""

    label: str
    score: float


@dataclass(frozen=True, config=ValidatorConfig)
class SegmentationMask(Annotation):
    """
    Raster segmentation mask. The ``locator`` is the URL to the image file representing the segmentation mask.

    The segmentation mask must be rendered as a single-channel, 8-bit-depth (grayscale) image. For the best results,
    use a lossless file format such as PNG. Each pixel's value is the numerical ID of its class label, as specified in
    the ``labels`` map. Any pixel value not present in the ``labels`` map is rendered as part of the background.

    For example, ``labels = {255: "object"}`` will highlight all pixels with the value of 255 as ``"object"``. Every
    other pixel value will be transparent.
    """

    #: Mapping of unique label IDs (pixel values) to unique label values.
    labels: Dict[int, str]

    #: URL of the image (segmentation mask).
    locator: str

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.SEGMENTATION_MASK


@dataclass(frozen=True, config=ValidatorConfig)
class BitmapMask(Annotation):
    """Arbitrary bitmap mask. The ``locator`` is the URL to the image file representing the mask."""

    #: URL of the bitmap data.
    locator: str

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.BITMAP_MASK


@dataclass(frozen=True, config=ValidatorConfig)
class ClassificationLabel(Annotation):
    """Label of classification."""

    label: str

    @staticmethod
    def _data_type() -> _AnnotationType:
        return _AnnotationType.CLASSIFICATON_LABEL


@dataclass(frozen=True, config=ValidatorConfig)
class ScoredClassificationLabel(ClassificationLabel):
    """Classification label with accompanying score."""

    score: float


_ANNOTATION_TYPES = [
    BoundingBox,
    LabeledBoundingBox,
    ScoredBoundingBox,
    ScoredLabeledBoundingBox,
    Polygon,
    LabeledPolygon,
    ScoredPolygon,
    ScoredLabeledPolygon,
    Keypoints,
    Polyline,
    BoundingBox3D,
    LabeledBoundingBox3D,
    ScoredBoundingBox3D,
    ScoredLabeledBoundingBox3D,
    SegmentationMask,
    BitmapMask,
    ClassificationLabel,
    ScoredClassificationLabel,
]
